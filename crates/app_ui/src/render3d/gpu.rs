//! The wgpu side of the 3D organ viewer: device setup, per-group buffers and
//! the three render passes described in `shaders.wgsl`.
//!
//! # Why this is not just "draw the triangles"
//!
//! Every organ but bone is translucent, and translucent geometry has to be
//! composited in depth order. The meshes interpenetrate — marching cubes over
//! neighbouring labels leaves shared walls and slivers — so no per-object sort
//! is correct, and a per-triangle sort of 75 000 triangles per frame is not
//! affordable in wasm. The viewer therefore draws opaque geometry first and
//! resolves the rest with weighted blended OIT, which is order-independent.
//!
//! WBOIT needs to *blend into a float colour attachment*. WebGPU guarantees
//! that for `Rgba16Float`; WebGL2 only offers it with `EXT_float_blend`, which
//! is not universal. [`Renderer::new`] probes for it and falls back to a plain
//! back-to-front sort by group centroid — wrong where two organs interpenetrate,
//! but the alternative is nothing at all.

use std::borrow::Cow;

use glam::Vec3;
use wgpu::util::DeviceExt;

use xelray_core::mesh3d::MeshBundle;

use super::camera::OrbitCamera;

/// Interleaved position + normal, `f32x3` each.
const VERTEX_STRIDE: wgpu::BufferAddress = 24;

/// Stride between per-group material slices in the shared uniform buffer.
///
/// WebGPU's `minUniformBufferOffsetAlignment` may be as coarse as 256 bytes,
/// so every slice is padded to that even though the payload is one `vec4`.
const MATERIAL_STRIDE: u32 = 256;

const DEPTH_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Depth24Plus;
const ACCUM_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::Rgba16Float;
const REVEAL_FORMAT: wgpu::TextureFormat = wgpu::TextureFormat::R16Float;

/// Mirrors `Globals` in `shaders.wgsl`.
#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Globals {
    view_proj: [[f32; 4]; 4],
    eye: [f32; 4],
    params: [f32; 4],
}

/// One mesh group's GPU residency, positionally matching
/// [`MeshBundle::groups`].
struct GpuGroup {
    vertices: wgpu::Buffer,
    indices: wgpu::Buffer,
    index_count: u32,
    /// `color[3] == 255` on disk. Opaque groups go through pass 1 and take no
    /// part in the transparency resolve.
    opaque: bool,
    /// Mean vertex position in *display* space, for the fallback sort.
    centroid: Vec3,
    /// Byte offset of this group's slice of the material buffer.
    material_offset: u32,
}

/// The two WBOIT attachments. Recreated with the surface, absent entirely on
/// the fallback path.
struct OitTargets {
    accum: wgpu::TextureView,
    reveal: wgpu::TextureView,
    /// Globals + the two targets, the composite pipeline's only bind group.
    bind_group: wgpu::BindGroup,
}

pub struct Renderer {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,

    depth: Option<wgpu::TextureView>,
    oit: Option<OitTargets>,

    globals_buffer: wgpu::Buffer,
    globals_bind_group: wgpu::BindGroup,
    material_bind_group: wgpu::BindGroup,
    /// Kept so [`Renderer::resize`] can rebuild the composite bind group.
    composite_layout: wgpu::BindGroupLayout,

    opaque_pipeline: wgpu::RenderPipeline,
    /// `Some` only when [`Renderer::wboit`] holds.
    wboit_pipeline: Option<wgpu::RenderPipeline>,
    composite_pipeline: Option<wgpu::RenderPipeline>,
    /// Sorted straight-alpha pipeline, the fallback for pass 2.
    blend_pipeline: Option<wgpu::RenderPipeline>,

    groups: Vec<GpuGroup>,
    /// Whether the adapter can blend into `Rgba16Float`.
    wboit: bool,
    /// The largest backing store this device will accept, so a 5K display
    /// cannot ask for a texture the driver refuses.
    max_dimension: u32,
    /// Scratch draw order for the fallback path, kept to avoid a per-frame
    /// allocation.
    order: Vec<usize>,
}

impl Renderer {
    /// Bring up a device on `canvas` and upload every group of `bundle`.
    ///
    /// The error is a plain string: nothing upstream can do anything with a
    /// richer type, and the only caller turns it into a message on screen.
    pub async fn new(
        canvas: web_sys::HtmlCanvasElement,
        bundle: &MeshBundle,
    ) -> Result<Renderer, String> {
        // `Instance::new` cannot detect WebGPU support synchronously, and an
        // instance created with BROWSER_WEBGPU set will refuse to fall back to
        // WebGL on its own. This helper does the async probe first.
        let instance = wgpu::util::new_instance_with_webgpu_detection(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::BROWSER_WEBGPU | wgpu::Backends::GL,
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        })
        .await;

        let width = canvas.width().max(1);
        let height = canvas.height().max(1);

        let surface = instance
            .create_surface(wgpu::SurfaceTarget::Canvas(canvas))
            .map_err(|e| format!("could not create a surface on the canvas: {e}"))?;

        // WebGL2 requires the surface up front — the adapter is the canvas's
        // context, not a free-standing device.
        let adapter = instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("no usable GPU adapter: {e}"))?;

        // Blending into a float colour attachment is core WebGPU but an
        // extension (`EXT_float_blend`) on WebGL2. Probing beats assuming.
        let accum_features = adapter.get_texture_format_features(ACCUM_FORMAT);
        let reveal_features = adapter.get_texture_format_features(REVEAL_FORMAT);
        let renderable = wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
        let wboit = accum_features.allowed_usages.contains(renderable)
            && reveal_features.allowed_usages.contains(renderable)
            && accum_features
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::BLENDABLE)
            && reveal_features
                .flags
                .contains(wgpu::TextureFormatFeatureFlags::BLENDABLE);

        // Ask for no optional features, and for the WebGL2 floor except in
        // resolution — the floor's 2048 px maximum is smaller than a retina
        // canvas on a large display.
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: Some("xelray-3d"),
                required_limits: wgpu::Limits::downlevel_webgl2_defaults()
                    .using_resolution(adapter.limits()),
                ..Default::default()
            })
            .await
            .map_err(|e| format!("could not open a GPU device: {e}"))?;

        let caps = surface.get_capabilities(&adapter);
        // Colours are converted to linear once on the CPU and lighting is done
        // in linear, so an sRGB surface — where the hardware encodes after
        // blending — is exactly right. Where none is offered the shader
        // encodes instead; see `encode()` there.
        let format = caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .or_else(|| caps.formats.first().copied())
            .ok_or_else(|| "the surface offers no texture format".to_string())?;

        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format,
            width,
            height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode: wgpu::CompositeAlphaMode::Opaque,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
            ..surface
                .get_default_config(&adapter, width, height)
                .ok_or_else(|| "the surface is not usable with this adapter".to_string())?
        };
        surface.configure(&device, &config);

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("xelray-3d"),
            source: wgpu::ShaderSource::Wgsl(Cow::Borrowed(include_str!("shaders.wgsl"))),
        });

        // ---- uniforms ------------------------------------------------------

        let globals_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("globals"),
            size: std::mem::size_of::<Globals>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let globals_entry = wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStages::VERTEX_FRAGMENT,
            ty: wgpu::BindingType::Buffer {
                ty: wgpu::BufferBindingType::Uniform,
                has_dynamic_offset: false,
                min_binding_size: None,
            },
            count: None,
        };

        let globals_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("globals"),
            entries: &[globals_entry],
        });
        let globals_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("globals"),
            layout: &globals_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: globals_buffer.as_entire_binding(),
            }],
        });

        // The material buffer is written once: a group's colour never changes,
        // only whether it is drawn. One dynamic offset per draw call selects a
        // slice.
        let mut material_bytes = vec![0u8; MATERIAL_STRIDE as usize * bundle.groups.len().max(1)];
        for (i, g) in bundle.groups.iter().enumerate() {
            let rgba = [
                srgb_to_linear(g.color[0]),
                srgb_to_linear(g.color[1]),
                srgb_to_linear(g.color[2]),
                g.color[3] as f32 / 255.0,
            ];
            let at = i * MATERIAL_STRIDE as usize;
            material_bytes[at..at + 16].copy_from_slice(bytemuck::bytes_of(&rgba));
        }
        let material_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("materials"),
            contents: &material_bytes,
            usage: wgpu::BufferUsages::UNIFORM,
        });

        let material_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("material"),
            entries: &[wgpu::BindGroupLayoutEntry {
                binding: 0,
                visibility: wgpu::ShaderStages::FRAGMENT,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: true,
                    min_binding_size: wgpu::BufferSize::new(16),
                },
                count: None,
            }],
        });
        let material_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("material"),
            layout: &material_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &material_buffer,
                    offset: 0,
                    size: wgpu::BufferSize::new(16),
                }),
            }],
        });

        // Group 0 for the composite pass: the same globals plus the two OIT
        // targets. See the binding comment in `shaders.wgsl`.
        let float_texture = wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        };
        let composite_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("composite"),
            entries: &[
                globals_entry,
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: float_texture,
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: float_texture,
                    count: None,
                },
            ],
        });

        // ---- pipelines -----------------------------------------------------

        let mesh_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("mesh"),
            bind_group_layouts: &[Some(&globals_layout), Some(&material_layout)],
            immediate_size: 0,
        });

        const VERTEX_ATTRS: [wgpu::VertexAttribute; 2] =
            wgpu::vertex_attr_array![0 => Float32x3, 1 => Float32x3];
        let vertex_buffers = [Some(wgpu::VertexBufferLayout {
            array_stride: VERTEX_STRIDE,
            step_mode: wgpu::VertexStepMode::Vertex,
            attributes: &VERTEX_ATTRS,
        })];

        let vertex = wgpu::VertexState {
            module: &shader,
            entry_point: Some("vs_mesh"),
            compilation_options: Default::default(),
            buffers: &vertex_buffers,
        };

        // Culling is off everywhere: the LPS -> display flip reverses winding,
        // and marching-cubes shells have slivers that vanish if either face is
        // dropped. The fragment shader flips back-facing normals instead.
        let primitive = wgpu::PrimitiveState {
            cull_mode: None,
            ..Default::default()
        };

        let depth_write = |write: bool| {
            Some(wgpu::DepthStencilState {
                format: DEPTH_FORMAT,
                depth_write_enabled: Some(write),
                depth_compare: Some(wgpu::CompareFunction::Less),
                stencil: Default::default(),
                bias: Default::default(),
            })
        };

        let opaque_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("opaque"),
            layout: Some(&mesh_layout),
            vertex: vertex.clone(),
            primitive,
            depth_stencil: depth_write(true),
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &shader,
                entry_point: Some("fs_opaque"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format,
                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview_mask: None,
            cache: None,
        });

        let (wboit_pipeline, composite_pipeline, blend_pipeline) = if wboit {
            let accumulate = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("wboit-accumulate"),
                layout: Some(&mesh_layout),
                vertex: vertex.clone(),
                primitive,
                // Tested against the opaque depth, never written: every
                // translucent fragment in front of the opaque image counts.
                depth_stencil: depth_write(false),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_wboit"),
                    compilation_options: Default::default(),
                    targets: &[
                        Some(wgpu::ColorTargetState {
                            format: ACCUM_FORMAT,
                            // Plain summation of weighted premultiplied colour.
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::One,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::One,
                                    dst_factor: wgpu::BlendFactor::One,
                                    operation: wgpu::BlendOperation::Add,
                                },
                            }),
                            write_mask: wgpu::ColorWrites::ALL,
                        }),
                        Some(wgpu::ColorTargetState {
                            format: REVEAL_FORMAT,
                            // dst = dst * (1 - a): the running product of what
                            // is still visible behind the layers so far.
                            blend: Some(wgpu::BlendState {
                                color: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::Zero,
                                    dst_factor: wgpu::BlendFactor::OneMinusSrc,
                                    operation: wgpu::BlendOperation::Add,
                                },
                                alpha: wgpu::BlendComponent {
                                    src_factor: wgpu::BlendFactor::Zero,
                                    dst_factor: wgpu::BlendFactor::OneMinusSrc,
                                    operation: wgpu::BlendOperation::Add,
                                },
                            }),
                            write_mask: wgpu::ColorWrites::RED,
                        }),
                    ],
                }),
                multiview_mask: None,
                cache: None,
            });

            let composite_pipeline_layout =
                device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: Some("composite"),
                    bind_group_layouts: &[Some(&composite_layout)],
                    immediate_size: 0,
                });

            let composite = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("wboit-composite"),
                layout: Some(&composite_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_fullscreen"),
                    compilation_options: Default::default(),
                    buffers: &[],
                },
                primitive,
                depth_stencil: None,
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_composite"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        // The canonical WBOIT resolve, with revealage in the
                        // alpha channel — see `fs_composite`.
                        blend: Some(wgpu::BlendState {
                            color: wgpu::BlendComponent {
                                src_factor: wgpu::BlendFactor::OneMinusSrcAlpha,
                                dst_factor: wgpu::BlendFactor::SrcAlpha,
                                operation: wgpu::BlendOperation::Add,
                            },
                            alpha: wgpu::BlendComponent::REPLACE,
                        }),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });

            (Some(accumulate), Some(composite), None)
        } else {
            let blend = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some("sorted-alpha"),
                layout: Some(&mesh_layout),
                vertex: vertex.clone(),
                primitive,
                depth_stencil: depth_write(false),
                multisample: Default::default(),
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_blend"),
                    compilation_options: Default::default(),
                    targets: &[Some(wgpu::ColorTargetState {
                        format,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                }),
                multiview_mask: None,
                cache: None,
            });
            (None, None, Some(blend))
        };

        // ---- geometry ------------------------------------------------------

        let groups: Vec<GpuGroup> = bundle
            .groups
            .iter()
            .enumerate()
            .map(|(i, g)| {
                let vert_count = g.positions.len() / 3;

                // Interleaved rather than two buffers: one vertex stream is
                // one binding, and the cache locality is free.
                let mut interleaved = Vec::with_capacity(vert_count * 6);
                let mut sum = Vec3::ZERO;
                for v in 0..vert_count {
                    let p = [g.positions[v * 3], g.positions[v * 3 + 1], g.positions[v * 3 + 2]];
                    interleaved.extend_from_slice(&p);
                    interleaved.extend_from_slice(
                        g.normals
                            .get(v * 3..v * 3 + 3)
                            // A group whose normals were truncated still
                            // draws; it is lit as if facing +z.
                            .unwrap_or(&[0.0, 0.0, 1.0]),
                    );
                    sum += Vec3::new(p[0], -p[1], p[2]);
                }
                let centroid = if vert_count == 0 {
                    Vec3::ZERO
                } else {
                    sum / vert_count as f32
                };

                GpuGroup {
                    vertices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&g.key),
                        contents: bytemuck::cast_slice(&interleaved),
                        usage: wgpu::BufferUsages::VERTEX,
                    }),
                    indices: device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some(&g.key),
                        contents: bytemuck::cast_slice(&g.indices),
                        usage: wgpu::BufferUsages::INDEX,
                    }),
                    index_count: g.indices.len() as u32,
                    opaque: g.color[3] == u8::MAX,
                    centroid,
                    material_offset: i as u32 * MATERIAL_STRIDE,
                }
            })
            .collect();

        let max_dimension = device.limits().max_texture_dimension_2d;
        let mut renderer = Renderer {
            surface,
            device,
            queue,
            config,
            depth: None,
            oit: None,
            globals_buffer,
            globals_bind_group,
            material_bind_group,
            composite_layout,
            opaque_pipeline,
            wboit_pipeline,
            composite_pipeline,
            blend_pipeline,
            order: Vec::with_capacity(groups.len()),
            groups,
            wboit,
            max_dimension,
        };
        renderer.rebuild_targets();
        Ok(renderer)
    }

    /// Whether the transparency resolve is the real one. Callers do not need
    /// this; it exists so a future overlay can say so.
    pub fn has_wboit(&self) -> bool {
        self.wboit
    }

    /// Reconfigure for a new backing-store size, in device pixels.
    pub fn resize(&mut self, width: u32, height: u32) {
        let width = width.clamp(1, self.max_dimension);
        let height = height.clamp(1, self.max_dimension);
        if (width, height) == (self.config.width, self.config.height) {
            return;
        }
        self.config.width = width;
        self.config.height = height;
        self.surface.configure(&self.device, &self.config);
        self.rebuild_targets();
    }

    /// Depth, and the OIT attachments, all of which follow the surface size.
    fn rebuild_targets(&mut self) {
        let size = wgpu::Extent3d {
            width: self.config.width,
            height: self.config.height,
            depth_or_array_layers: 1,
        };
        let target = |label, format, usage| {
            self.device
                .create_texture(&wgpu::TextureDescriptor {
                    label: Some(label),
                    size,
                    mip_level_count: 1,
                    sample_count: 1,
                    dimension: wgpu::TextureDimension::D2,
                    format,
                    usage,
                    view_formats: &[],
                })
                .create_view(&Default::default())
        };

        self.depth = Some(target(
            "depth",
            DEPTH_FORMAT,
            wgpu::TextureUsages::RENDER_ATTACHMENT,
        ));

        self.oit = if self.wboit {
            let usage =
                wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING;
            let accum = target("oit-accum", ACCUM_FORMAT, usage);
            let reveal = target("oit-reveal", REVEAL_FORMAT, usage);
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("composite"),
                layout: &self.composite_layout,
                entries: &[
                    wgpu::BindGroupEntry {
                        binding: 0,
                        resource: self.globals_buffer.as_entire_binding(),
                    },
                    wgpu::BindGroupEntry {
                        binding: 1,
                        resource: wgpu::BindingResource::TextureView(&accum),
                    },
                    wgpu::BindGroupEntry {
                        binding: 2,
                        resource: wgpu::BindingResource::TextureView(&reveal),
                    },
                ],
            });
            Some(OitTargets {
                accum,
                reveal,
                bind_group,
            })
        } else {
            None
        };
    }

    /// Draw one frame. `visible[i]` gates group `i`; a short slice hides the
    /// rest.
    pub fn render(&mut self, camera: &OrbitCamera, visible: &[bool]) {
        let frame = match self.surface.get_current_texture() {
            wgpu::CurrentSurfaceTexture::Success(t)
            | wgpu::CurrentSurfaceTexture::Suboptimal(t) => t,
            // Outdated/Lost mean the canvas changed size under us; the resize
            // that caused it schedules its own redraw, so dropping this frame
            // is the whole recovery.
            _ => return,
        };
        let view = frame.texture.create_view(&Default::default());
        let Some(depth) = self.depth.as_ref() else {
            return;
        };

        let aspect = self.config.width as f32 / self.config.height.max(1) as f32;
        let eye = camera.eye();
        self.queue.write_buffer(
            &self.globals_buffer,
            0,
            bytemuck::bytes_of(&Globals {
                view_proj: camera.view_proj(aspect).to_cols_array_2d(),
                eye: [eye.x, eye.y, eye.z, 0.0],
                params: [if self.config.format.is_srgb() { 0.0 } else { 1.0 }, 0.0, 0.0, 0.0],
            }),
        );

        let drawn = |i: usize, g: &GpuGroup| {
            g.index_count > 0 && visible.get(i).copied().unwrap_or(false)
        };
        let any_translucent = self
            .groups
            .iter()
            .enumerate()
            .any(|(i, g)| !g.opaque && drawn(i, g));

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: Some("frame") });

        // ---- pass 1: opaque ------------------------------------------------
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("opaque"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &view,
                    depth_slice: None,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::BLACK),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                ..Default::default()
            });
            pass.set_pipeline(&self.opaque_pipeline);
            pass.set_bind_group(0, &self.globals_bind_group, &[]);
            for (i, g) in self.groups.iter().enumerate() {
                if g.opaque && drawn(i, g) {
                    draw_group(&mut pass, &self.material_bind_group, g);
                }
            }
        }

        if any_translucent {
            match (self.wboit_pipeline.as_ref(), self.oit.as_ref()) {
                (Some(accumulate), Some(oit)) => {
                    // ---- pass 2: WBOIT accumulate ----------------------------
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("wboit-accumulate"),
                        color_attachments: &[
                            Some(wgpu::RenderPassColorAttachment {
                                view: &oit.accum,
                                depth_slice: None,
                                resolve_target: None,
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                            Some(wgpu::RenderPassColorAttachment {
                                view: &oit.reveal,
                                depth_slice: None,
                                resolve_target: None,
                                // Revealage starts at "nothing hidden yet".
                                ops: wgpu::Operations {
                                    load: wgpu::LoadOp::Clear(wgpu::Color::WHITE),
                                    store: wgpu::StoreOp::Store,
                                },
                            }),
                        ],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                    pass.set_pipeline(accumulate);
                    pass.set_bind_group(0, &self.globals_bind_group, &[]);
                    for (i, g) in self.groups.iter().enumerate() {
                        if !g.opaque && drawn(i, g) {
                            draw_group(&mut pass, &self.material_bind_group, g);
                        }
                    }
                    drop(pass);

                    // ---- pass 3: composite -----------------------------------
                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("wboit-composite"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: None,
                        ..Default::default()
                    });
                    if let Some(composite) = self.composite_pipeline.as_ref() {
                        pass.set_pipeline(composite);
                        pass.set_bind_group(0, &oit.bind_group, &[]);
                        pass.draw(0..3, 0..1);
                    }
                }
                _ => {
                    // ---- fallback: sorted alpha ------------------------------
                    // Back to front by group centroid. Correct for organs that
                    // do not interpenetrate, which is most of them.
                    self.order.clear();
                    self.order.extend(
                        self.groups
                            .iter()
                            .enumerate()
                            .filter(|(i, g)| !g.opaque && drawn(*i, g))
                            .map(|(i, _)| i),
                    );
                    let key = |i: &usize| (self.groups[*i].centroid - eye).length_squared();
                    self.order
                        .sort_by(|a, b| key(b).total_cmp(&key(a)));

                    let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                        label: Some("sorted-alpha"),
                        color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                            view: &view,
                            depth_slice: None,
                            resolve_target: None,
                            ops: wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            },
                        })],
                        depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                            view: depth,
                            depth_ops: Some(wgpu::Operations {
                                load: wgpu::LoadOp::Load,
                                store: wgpu::StoreOp::Store,
                            }),
                            stencil_ops: None,
                        }),
                        ..Default::default()
                    });
                    if let Some(blend) = self.blend_pipeline.as_ref() {
                        pass.set_pipeline(blend);
                        pass.set_bind_group(0, &self.globals_bind_group, &[]);
                        for i in &self.order {
                            draw_group(&mut pass, &self.material_bind_group, &self.groups[*i]);
                        }
                    }
                }
            }
        }

        self.queue.submit(Some(encoder.finish()));
        self.queue.present(frame);
    }
}

fn draw_group(pass: &mut wgpu::RenderPass<'_>, material: &wgpu::BindGroup, g: &GpuGroup) {
    pass.set_bind_group(1, material, &[g.material_offset]);
    pass.set_vertex_buffer(0, g.vertices.slice(..));
    pass.set_index_buffer(g.indices.slice(..), wgpu::IndexFormat::Uint32);
    pass.draw_indexed(0..g.index_count, 0, 0..1);
}

/// sRGB byte to linear float. Done once at upload, so lighting and blending
/// both run in linear light.
fn srgb_to_linear(c: u8) -> f32 {
    let c = c as f32 / 255.0;
    if c <= 0.040_45 {
        c / 12.92
    } else {
        ((c + 0.055) / 1.055).powf(2.4)
    }
}
