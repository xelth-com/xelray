//! 3D organ-mesh viewer (wgpu): the camera, the GPU renderer, and the Leptos
//! component that glues them to a canvas.
//!
//! The component renders **on demand**. A segmentation is static geometry —
//! nothing moves unless the user moves it — so a free-running
//! `requestAnimationFrame` loop would burn a laptop battery to redraw the same
//! pixels sixty times a second. Every gesture, every visibility change and
//! every resize instead asks for exactly one frame, coalesced through a
//! pending flag.
//!
//! # Why the target gate
//!
//! [`gpu`] and [`stage`] are wasm-only: wgpu's canvas surface, and half the
//! events they bind, simply do not exist off the web. The crate is still built
//! for the host to run its unit tests, so `Stage3d` has a do-nothing twin
//! there and `lib.rs` needs no `cfg` of its own.

pub mod camera;

#[cfg(target_arch = "wasm32")]
pub mod gpu;

#[cfg(target_arch = "wasm32")]
mod stage;
#[cfg(target_arch = "wasm32")]
pub use stage::Stage3d;

#[cfg(not(target_arch = "wasm32"))]
#[leptos::component]
pub fn Stage3d(
    bundle: std::rc::Rc<xelray_core::mesh3d::MeshBundle>,
    organ_visible: leptos::RwSignal<u32>,
    cam_reset: leptos::RwSignal<u64>,
    i18n: crate::i18n::I18n,
) -> impl leptos::IntoView {
    let _ = (bundle, organ_visible, cam_reset, i18n);
    leptos::view! { <div class="stage3d"></div> }
}

#[cfg(test)]
mod tests {
    /// The shaders are only ever compiled inside a browser, where a typo shows
    /// up as a blank canvas and a console message no CI run will ever read.
    /// naga is the same front end wgpu uses, so parsing and validating the
    /// file here catches the same errors natively.
    #[test]
    fn shaders_are_valid_wgsl() {
        let source = include_str!("shaders.wgsl");
        let module = naga::front::wgsl::parse_str(source).expect("shaders.wgsl should parse");

        let mut validator = naga::valid::Validator::new(
            naga::valid::ValidationFlags::all(),
            naga::valid::Capabilities::default(),
        );
        validator
            .validate(&module)
            .expect("shaders.wgsl should validate");
    }
}
