//! The `Stage3d` component: a canvas, a wgpu [`Renderer`], and the gestures
//! that drive the camera. See the parent module for the render-on-demand
//! reasoning.

use std::cell::{Cell, RefCell};
use std::rc::Rc;

use leptos::html::Canvas;
use leptos::*;
use wasm_bindgen::prelude::*;

use xelray_core::mesh3d::{MeshBundle, MeshGroup};

use crate::i18n::I18n;

use super::camera::OrbitCamera;
use super::gpu::Renderer;

/// Accumulated wheel pixels worth one doubling of the distance to the target.
/// Continuous rather than stepped, so a trackpad pinch feels analogue and a
/// mouse detent still moves a sensible amount — the same treatment the 2D
/// stage gives zoom.
const PIXELS_PER_ZOOM_DOUBLING: f64 = 260.0;

/// Backing-store scale is capped here. Past 2x the extra pixels are invisible
/// and the fill cost is not.
const MAX_PIXEL_RATIO: f64 = 2.0;

/// Ceiling on the backing store, in device pixels.
///
/// Every frame is three full-screen passes over a `Depth24Plus`, an
/// `Rgba16Float` and an `R16Float` attachment — about 18 bytes of traffic per
/// pixel per frame. A maximised window on a 4K panel at [`MAX_PIXEL_RATIO`]
/// asks for 33 megapixels of that, which no integrated GPU finishes inside a
/// frame; the drag then falls behind by whole seconds because every `rAF`
/// costs more than the interval between them. 4 Mpx (about 2440x1640) is past
/// the point the extra samples show and safely inside a frame budget, so the
/// ratio is scaled down rather than the window being asked to be smaller.
const MAX_BACKING_PIXELS: f64 = 4_000_000.0;

/// What the current pointer drag is doing.
#[derive(Clone, Copy, PartialEq)]
enum Drag {
    Orbit,
    Pan,
}

/// Normalise a wheel delta to CSS pixels. Firefox reports lines (`deltaMode`
/// 1) and, rarely, pages (2).
fn wheel_pixels(ev: &web_sys::WheelEvent) -> f64 {
    match ev.delta_mode() {
        1 => ev.delta_y() * 16.0,
        2 => ev.delta_y() * 400.0,
        _ => ev.delta_y(),
    }
}

/// Axis-aligned bounds of one group's vertices, in LPS millimetres, or `None`
/// for a group that holds no vertices.
///
/// [`MeshBundle::bbox`] spans every group at once, which is the wrong box to
/// frame: bones run from the ribs to the pelvis and are hidden by default, so
/// fitting to the whole bundle leaves the abdominal organs small and off
/// centre. Computing these once at mount is a single pass over ~75k vertices —
/// cheaper than the buffer upload beside it.
fn group_bbox(g: &MeshGroup) -> Option<([f32; 3], [f32; 3])> {
    let mut points = g.positions.chunks_exact(3);
    let first = points.next()?;
    let mut min = [first[0], first[1], first[2]];
    let mut max = min;
    for p in points {
        for a in 0..3 {
            min[a] = min[a].min(p[a]);
            max[a] = max[a].max(p[a]);
        }
    }
    Some((min, max))
}

/// The box the camera should frame: the union of the groups `mask` shows.
///
/// Falls back to `full` — the bundle's own box — when the mask leaves nothing
/// visible, so recentring with every organ switched off still points the
/// camera at the anatomy rather than at the origin.
fn shown_bbox(
    boxes: &[Option<([f32; 3], [f32; 3])>],
    mask: u32,
    full: ([f32; 3], [f32; 3]),
) -> ([f32; 3], [f32; 3]) {
    let mut acc: Option<([f32; 3], [f32; 3])> = None;
    for (i, group) in boxes.iter().enumerate() {
        // The mask is 32 bits wide; a longer bundle's tail is never visible.
        if i >= 32 || mask >> i & 1 == 0 {
            continue;
        }
        let Some((gmin, gmax)) = *group else { continue };
        match &mut acc {
            None => acc = Some((gmin, gmax)),
            Some((min, max)) => {
                for a in 0..3 {
                    min[a] = min[a].min(gmin[a]);
                    max[a] = max[a].max(gmax[a]);
                }
            }
        }
    }
    acc.unwrap_or(full)
}

/// The pinch distance and midpoint of the first two touches.
fn pinch_state(ev: &web_sys::TouchEvent) -> Option<(f64, f64, f64)> {
    let touches = ev.touches();
    let (a, b) = (touches.get(0)?, touches.get(1)?);
    let (ax, ay) = (a.client_x() as f64, a.client_y() as f64);
    let (bx, by) = (b.client_x() as f64, b.client_y() as f64);
    Some((
        ((bx - ax).powi(2) + (by - ay).powi(2)).sqrt(),
        (ax + bx) * 0.5,
        (ay + by) * 0.5,
    ))
}

/// The 3D stage: a canvas filling its container, plus every gesture that
/// drives the camera.
///
/// `organ_visible` is a bitmask over [`MeshBundle::groups`] — bit `n` shows
/// group `n`. Bumping `cam_reset` re-frames the groups that mask currently
/// shows. Both are owned by the caller, so the organ rail beside this drives
/// them without reaching into the component.
///
/// `i18n` is taken by value rather than read from context because the only
/// string here is a failure message, and a component that renders one line of
/// text should not need a context lookup to do it.
#[component]
pub fn Stage3d(
    bundle: Rc<MeshBundle>,
    organ_visible: RwSignal<u32>,
    cam_reset: RwSignal<u64>,
    /// Brightness multiplier on the lit colour; `1.0` is the designed look.
    exposure: RwSignal<f32>,
    i18n: I18n,
) -> impl IntoView {
    let canvas_ref = create_node_ref::<Canvas>();

    let group_count = bundle.groups.len();
    let renderer: Rc<RefCell<Option<Renderer>>> = Rc::new(RefCell::new(None));

    // Framing follows what is on screen, not what is in the file — see
    // `group_bbox`. The same closure serves the mount and every recentre, so
    // `r` re-fits to whatever is showing now instead of restoring a pose
    // chosen when a different set of organs was visible.
    let fit_visible: Rc<dyn Fn() -> OrbitCamera> = {
        let boxes: Rc<Vec<_>> = Rc::new(bundle.groups.iter().map(group_bbox).collect());
        let full = bundle.bbox;
        Rc::new(move || {
            let (min, max) = shown_bbox(&boxes, organ_visible.get_untracked(), full);
            OrbitCamera::fit(min, max)
        })
    };

    let camera = Rc::new(RefCell::new(fit_visible()));
    // A frame is already queued; further requests fold into it.
    let pending = Rc::new(Cell::new(false));
    // Reused rather than rebuilt each frame: the renderer only reads it.
    let visible = Rc::new(RefCell::new(vec![false; group_count]));
    // Viewport height in CSS pixels, which `pan` needs to convert a drag into
    // a world-space offset.
    let css_height = Rc::new(Cell::new(1.0f64));
    let failure = create_rw_signal::<Option<String>>(None);

    let schedule: Rc<dyn Fn()> = {
        let renderer = renderer.clone();
        let camera = camera.clone();
        let pending = pending.clone();
        let visible = visible.clone();
        Rc::new(move || {
            if pending.replace(true) {
                return;
            }
            let (renderer, camera, pending, visible) = (
                renderer.clone(),
                camera.clone(),
                pending.clone(),
                visible.clone(),
            );
            request_animation_frame(move || {
                // Cleared first, so a gesture that arrives while this frame is
                // being recorded queues the *next* one instead of being
                // dropped. At most one frame is ever outstanding either way.
                pending.set(false);
                let mask = organ_visible.get_untracked();
                {
                    let mut visible = visible.borrow_mut();
                    for (i, shown) in visible.iter_mut().enumerate() {
                        *shown = i < 32 && mask >> i & 1 == 1;
                    }
                }
                if let Some(r) = renderer.borrow_mut().as_mut() {
                    r.render(&camera.borrow(), &visible.borrow(), exposure.get_untracked());
                }
            });
        })
    };

    // Toggling an organ, nudging the brightness or resetting the camera is a
    // redraw and nothing else.
    {
        let schedule = schedule.clone();
        create_effect(move |_| {
            organ_visible.track();
            exposure.track();
            schedule();
        });
    }
    {
        let schedule = schedule.clone();
        let camera = camera.clone();
        let fit_visible = fit_visible.clone();
        create_effect(move |seen: Option<u64>| {
            let n = cam_reset.get();
            // The first run is the mount, not a reset request.
            if seen.is_some() {
                // Re-fit rather than restore: after hiding the bones the pose
                // the mount chose is the wrong framing, and "recentre" that
                // leaves the model off centre teaches the key means nothing.
                *camera.borrow_mut() = fit_visible();
                schedule();
            }
            n
        });
    }

    // The observer has to be disconnected by hand: it holds the canvas alive
    // and would keep firing into disposed signals after the study is closed.
    let observer = Rc::new(RefCell::new(None::<web_sys::ResizeObserver>));

    {
        let renderer = renderer.clone();
        let camera = camera.clone();
        let schedule = schedule.clone();
        let observer = observer.clone();
        let css_height = css_height.clone();

        canvas_ref.on_load(move |el| {
            let canvas: web_sys::HtmlCanvasElement = (*el).clone();

            // ---- size ------------------------------------------------------
            // The backing store is CSS pixels times the display's ratio; the
            // element itself is sized by CSS. Doing this before the device is
            // created saves an immediate reconfigure.
            let resize = {
                let canvas = canvas.clone();
                let renderer = renderer.clone();
                let schedule = schedule.clone();
                let css_height = css_height.clone();
                move || {
                    let mut ratio = window().device_pixel_ratio().clamp(1.0, MAX_PIXEL_RATIO);
                    let w = canvas.client_width().max(1) as f64;
                    let h = canvas.client_height().max(1) as f64;
                    css_height.set(h);
                    // Trade sharpness for fill rate only where the frame would
                    // otherwise not fit in its budget — see MAX_BACKING_PIXELS.
                    let budget = MAX_BACKING_PIXELS / (w * h);
                    if budget < ratio * ratio {
                        ratio = budget.sqrt().max(1.0);
                    }
                    let (bw, bh) = ((w * ratio) as u32, (h * ratio) as u32);
                    if canvas.width() == bw && canvas.height() == bh {
                        // The observer fires for sub-pixel layout changes and
                        // once more after this callback writes the canvas's
                        // own attributes. Neither is a new frame, and a
                        // redundant `schedule` here is a redundant repaint.
                        return;
                    }
                    canvas.set_width(bw);
                    canvas.set_height(bh);
                    if let Some(r) = renderer.borrow_mut().as_mut() {
                        r.resize(bw, bh);
                    }
                    schedule();
                }
            };
            resize();

            let closure = Closure::wrap(Box::new({
                let resize = resize.clone();
                move |_: js_sys::Array| resize()
            }) as Box<dyn FnMut(js_sys::Array)>);
            if let Ok(obs) = web_sys::ResizeObserver::new(closure.as_ref().unchecked_ref()) {
                // The pane, not the canvas: the callback writes the canvas's
                // intrinsic size, and an observer watching the element it
                // resizes is the classic ResizeObserver feedback loop. The
                // pane is sized by CSS alone and nothing here ever touches it.
                let pane = canvas.parent_element();
                obs.observe(pane.as_ref().unwrap_or(canvas.unchecked_ref()));
                *observer.borrow_mut() = Some(obs);
            }
            // The observer outlives this call and owns the callback.
            closure.forget();

            // ---- device ----------------------------------------------------
            {
                let canvas = canvas.clone();
                let renderer = renderer.clone();
                let schedule = schedule.clone();
                let bundle = bundle.clone();
                spawn_local(async move {
                    match Renderer::new(canvas, &bundle).await {
                        Ok(r) => {
                            *renderer.borrow_mut() = Some(r);
                            schedule();
                        }
                        Err(e) => failure.set(Some(e)),
                    }
                });
            }

            // ---- gestures --------------------------------------------------
            // Bound straight to the element rather than through Leptos, whose
            // delegated listeners are passive and cannot `preventDefault` a
            // wheel or a touch. See `viewport::listen_active`.
            install_gestures(&canvas, camera.clone(), schedule.clone(), css_height.clone());
        });
    }

    on_cleanup(move || {
        if let Some(obs) = observer.borrow_mut().take() {
            obs.disconnect();
        }
        // Release the device and every vertex buffer with it; a study can be
        // closed and another opened without the first one's 3.5 MB lingering.
        renderer.borrow_mut().take();
    });

    view! {
        // Sized entirely by `.stage3d` in the stylesheet: it claims the row
        // the rail leaves, and the canvas is absolutely positioned inside it
        // so the element never grows itself from its own backing-store size.
        <div class="stage3d">
            <canvas node_ref=canvas_ref></canvas>
            <Show when=move || failure.get().is_some()>
                // Dressed as the slice viewer's own failure card, so a browser
                // that cannot do either one says so the same way twice.
                <div class="stage-message error">
                    <div class="sm-body">{move || i18n.t("xelray.viewer3d.no_gpu")}</div>
                </div>
            </Show>
        </div>
    }
}

/// Wire pointer, wheel and touch gestures to `camera`, redrawing after each.
///
/// Split out only to keep [`Stage3d`] readable; the camera math itself all
/// lives in [`camera`].
fn install_gestures(
    canvas: &web_sys::HtmlCanvasElement,
    camera: Rc<RefCell<OrbitCamera>>,
    schedule: Rc<dyn Fn()>,
    css_height: Rc<Cell<f64>>,
) {
    use crate::viewport::listen_active;

    // `(kind, last_x, last_y)`, or `None` when no button is down.
    let drag = Rc::new(Cell::new(None::<(Drag, f64, f64)>));
    // One-finger touch: the previous position.
    let touch = Rc::new(Cell::new(None::<(f64, f64)>));
    // Two-finger touch: pinch distance and midpoint at the last move.
    let pinch = Rc::new(Cell::new(None::<(f64, f64, f64)>));

    let target: &web_sys::EventTarget = canvas;

    {
        let drag = drag.clone();
        let canvas = canvas.clone();
        listen_active(target, "pointerdown", move |ev: web_sys::PointerEvent| {
            let kind = match ev.button() {
                0 => Drag::Orbit,
                // Right-drag pans, and the context menu is suppressed below.
                2 => Drag::Pan,
                _ => return,
            };
            ev.prevent_default();
            // Capture keeps the drag alive when the pointer leaves the canvas,
            // which orbiting past the edge does constantly.
            let _ = canvas.set_pointer_capture(ev.pointer_id());
            drag.set(Some((kind, ev.client_x() as f64, ev.client_y() as f64)));
        });
    }

    {
        let drag = drag.clone();
        let camera = camera.clone();
        let schedule = schedule.clone();
        let css_height = css_height.clone();
        listen_active(target, "pointermove", move |ev: web_sys::PointerEvent| {
            let Some((kind, last_x, last_y)) = drag.get() else {
                return;
            };
            let (x, y) = (ev.client_x() as f64, ev.client_y() as f64);
            drag.set(Some((kind, x, y)));
            let (dx, dy) = ((x - last_x) as f32, (y - last_y) as f32);

            match kind {
                Drag::Orbit => camera.borrow_mut().orbit(dx, dy),
                Drag::Pan => camera
                    .borrow_mut()
                    .pan(dx, dy, css_height.get() as f32),
            }
            schedule();
        });
    }

    for name in ["pointerup", "pointercancel"] {
        let drag = drag.clone();
        let canvas = canvas.clone();
        listen_active(target, name, move |ev: web_sys::PointerEvent| {
            drag.set(None);
            let _ = canvas.release_pointer_capture(ev.pointer_id());
        });
    }

    // Right-drag is a pan, so the menu it would otherwise open has to go.
    listen_active(target, "contextmenu", |ev: web_sys::Event| {
        ev.prevent_default();
    });

    {
        let camera = camera.clone();
        let schedule = schedule.clone();
        listen_active(target, "wheel", move |ev: web_sys::WheelEvent| {
            ev.prevent_default();
            // One exponential for both a trackpad's stream of small deltas and
            // a mouse's single large one: the 2D stage has to tell them apart
            // because it steps through discrete slices, but a continuous zoom
            // does the right thing either way.
            let factor = (wheel_pixels(&ev) / PIXELS_PER_ZOOM_DOUBLING).exp2();
            camera.borrow_mut().zoom(factor as f32);
            schedule();
        });
    }

    // ---- touch -------------------------------------------------------------

    {
        let touch = touch.clone();
        let pinch = pinch.clone();
        listen_active(target, "touchstart", move |ev: web_sys::TouchEvent| {
            ev.prevent_default();
            match ev.touches().length() {
                1 => {
                    let t = ev.touches().get(0);
                    touch.set(t.map(|t| (t.client_x() as f64, t.client_y() as f64)));
                    pinch.set(None);
                }
                2 => {
                    touch.set(None);
                    pinch.set(pinch_state(&ev));
                }
                _ => {
                    touch.set(None);
                    pinch.set(None);
                }
            }
        });
    }

    {
        let touch = touch.clone();
        let pinch = pinch.clone();
        let camera = camera.clone();
        let schedule = schedule.clone();
        let css_height = css_height.clone();
        listen_active(target, "touchmove", move |ev: web_sys::TouchEvent| {
            ev.prevent_default();

            if let Some((last_d, last_cx, last_cy)) = pinch.get() {
                let Some((d, cx, cy)) = pinch_state(&ev) else {
                    return;
                };
                pinch.set(Some((d, cx, cy)));
                // Spreading the fingers moves the eye closer.
                if last_d > 1.0 && d > 1.0 {
                    camera.borrow_mut().zoom((last_d / d) as f32);
                }
                // The midpoint travelling is a pan, so both happen at once.
                camera.borrow_mut().pan(
                    (cx - last_cx) as f32,
                    (cy - last_cy) as f32,
                    css_height.get() as f32,
                );
                schedule();
                return;
            }

            if let Some((last_x, last_y)) = touch.get() {
                let Some(t) = ev.touches().get(0) else { return };
                let (x, y) = (t.client_x() as f64, t.client_y() as f64);
                touch.set(Some((x, y)));
                camera
                    .borrow_mut()
                    .orbit((x - last_x) as f32, (y - last_y) as f32);
                schedule();
            }
        });
    }

    for name in ["touchend", "touchcancel"] {
        let touch = touch.clone();
        let pinch = pinch.clone();
        listen_active(target, name, move |ev: web_sys::TouchEvent| {
            // Lifting one of two fingers must not make the remaining one jump:
            // the next `touchstart`/`touchmove` re-seeds from scratch.
            let _ = ev;
            touch.set(None);
            pinch.set(None);
        });
    }
}
