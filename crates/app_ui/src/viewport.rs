//! The image stage: canvas plus the pointer, wheel and touch gestures.
//!
//! Zoom and pan are a CSS transform on the canvas element rather than a
//! redraw, so panning a 512² CT costs nothing — the pixels are only
//! recomputed when the slice or the window changes. The canvas is fitted to
//! the pane by `max-width`/`max-height`, so it follows a window resize with
//! no JavaScript at all.
//!
//! There is no pan *mode*. At fitted size there is nothing to pan, so a drag
//! sets brightness; once zoomed in, dragging moves the image, which is what
//! dragging a too-large picture does everywhere else, and `Shift` gets
//! brightness back. Middle-drag always moves.

use leptos::html::{Canvas, Div};
use leptos::*;
use wasm_bindgen::prelude::*;
use web_sys::{AddEventListenerOptions, Event};

use crate::Viewer;

/// What the current drag is doing.
#[derive(Clone, Copy, PartialEq)]
enum Drag {
    /// Horizontal adjusts window width, vertical adjusts window level.
    Window,
    Pan,
}

/// Roughly how many modality units one pixel of drag is worth.
const WINDOW_SENSITIVITY: f64 = 2.0;

/// A wheel event bigger than this is a notched mouse wheel, not a trackpad.
///
/// Mouse wheels emit one large delta per detent; trackpads emit a stream of
/// small ones. Treating them the same makes a two-finger swipe skip dozens
/// of slices, so they are told apart by magnitude.
const NOTCH_THRESHOLD: f64 = 45.0;

/// Accumulated trackpad scroll, in CSS pixels, worth one slice.
const PIXELS_PER_SLICE: f64 = 24.0;

/// Accumulated pixels worth one doubling of the zoom, for pinch and
/// ctrl+wheel. Continuous rather than stepped, so pinch feels analogue.
const PIXELS_PER_ZOOM_DOUBLING: f64 = 260.0;

/// Swipe distance worth one slice, for one-finger touch.
const TOUCH_PIXELS_PER_SLICE: f64 = 18.0;

/// Attach a listener that is allowed to call `preventDefault`.
///
/// Leptos delegates `on:` handlers to the document, and Chrome forces
/// document-level `wheel` and `touch*` listeners passive — which silently
/// drops our `preventDefault` and fills the console with intervention
/// warnings. Binding straight to the element with `passive: false` is the
/// only way to own these gestures.
pub(crate) fn listen_active<E>(
    el: &web_sys::EventTarget,
    name: &str,
    mut handler: impl FnMut(E) + 'static,
)
where
    E: JsCast,
{
    let closure = Closure::wrap(Box::new(move |ev: Event| {
        handler(ev.unchecked_into::<E>());
    }) as Box<dyn FnMut(Event)>);

    let options = AddEventListenerOptions::new();
    options.set_passive(false);
    let _ = el.add_event_listener_with_callback_and_add_event_listener_options(
        name,
        closure.as_ref().unchecked_ref(),
        &options,
    );
    // The stage lives for as long as the study is open and the listener must
    // outlive this call; leaking one closure per gesture is the price.
    closure.forget();
}

/// Normalise a wheel delta to CSS pixels.
///
/// Firefox reports lines (`deltaMode` 1) and, rarely, pages (2).
fn wheel_pixels(ev: &ev::WheelEvent) -> f64 {
    match ev.delta_mode() {
        1 => ev.delta_y() * 16.0,
        2 => ev.delta_y() * 400.0,
        _ => ev.delta_y(),
    }
}

#[component]
pub fn Stage(v: Viewer, canvas_ref: NodeRef<Canvas>) -> impl IntoView {
    let stage_ref = create_node_ref::<Div>();

    // `(kind, last_x, last_y)` — deltas are taken against the previous move
    // so the gesture keeps working when the pointer leaves the element.
    let drag = create_rw_signal::<Option<(Drag, f64, f64)>>(None);
    // Leftover scroll distance that has not yet added up to a whole slice.
    let scroll_acc = store_value(0.0f64);
    // One-finger touch: last position, accumulated swipe.
    let touch = store_value::<Option<(f64, f64, f64)>>(None);
    // Two-finger touch: pinch distance and centroid Y at the last move.
    let pinch = store_value::<Option<(f64, f64, f64)>>(None);

    // These live on `window` so a drag keeps working when the pointer leaves
    // the image — but they therefore outlive this component unless removed.
    // Dropping the handle does *not* unregister them: closing a study would
    // leave them firing against disposed signals, which aborts the module on
    // the next mouse move. Hence the explicit cleanup below.
    let on_move = window_event_listener(ev::mousemove, move |ev| {
        let Some((kind, last_x, last_y)) = drag.get_untracked() else {
            return;
        };
        let (x, y) = (ev.client_x() as f64, ev.client_y() as f64);
        let (dx, dy) = (x - last_x, y - last_y);
        drag.set(Some((kind, x, y)));

        match kind {
            Drag::Window => {
                v.ww.update(|w| *w = (*w + dx * WINDOW_SENSITIVITY).max(1.0));
                // Dragging up raises the level, the radiology convention.
                v.wl.update(|l| *l -= dy * WINDOW_SENSITIVITY);
            }
            Drag::Pan => v.pan.update(|(px, py)| {
                *px += dx;
                *py += dy;
            }),
        }
    });

    let on_up = window_event_listener(ev::mouseup, move |_| drag.set(None));

    on_cleanup(move || {
        on_move.remove();
        on_up.remove();
    });

    let on_mousedown = move |ev: ev::MouseEvent| {
        let kind = match ev.button() {
            // The middle button is the universal "grab and move".
            1 => Drag::Pan,
            // Zoomed in, dragging moves the image the way dragging any
            // oversized picture does; Shift asks for brightness instead.
            0 if v.is_zoomed() && !ev.shift_key() => Drag::Pan,
            0 => Drag::Window,
            _ => return,
        };
        ev.prevent_default();
        drag.set(Some((kind, ev.client_x() as f64, ev.client_y() as f64)));
    };

    let on_wheel = move |ev: ev::WheelEvent| {
        ev.prevent_default();
        let delta = wheel_pixels(&ev);

        // A trackpad pinch reaches the page as ctrl+wheel, so this one
        // branch covers both pinching and ctrl+scrolling. Zoom is applied
        // as a continuous exponential rather than in steps, which keeps a
        // pinch smooth and still moves a mouse notch a sensible amount.
        if ev.ctrl_key() || ev.meta_key() {
            v.zoom.update(|z| {
                let factor = (-delta / PIXELS_PER_ZOOM_DOUBLING).exp2();
                *z = (*z * factor).clamp(0.1, 12.0);
            });
            return;
        }

        // Any manual step through the stack stops cine playback.
        v.pause_cine();

        if delta.abs() >= NOTCH_THRESHOLD {
            // A real mouse wheel: one detent, one slice, no carry-over —
            // otherwise the remainder makes the occasional notch jump two.
            scroll_acc.set_value(0.0);
            v.step_slice(if delta > 0.0 { 1 } else { -1 });
            return;
        }

        let acc = scroll_acc.get_value() + delta;
        let steps = (acc / PIXELS_PER_SLICE).trunc();
        scroll_acc.set_value(acc - steps * PIXELS_PER_SLICE);
        if steps != 0.0 {
            v.step_slice(steps as i32);
        }
    };

    // ---- touch ------------------------------------------------------------
    // One finger swipes through the stack, or moves the image once it is
    // zoomed in — the same rule as the mouse. Two fingers pinch to zoom and
    // scroll to step, both at once.

    let on_touchstart = move |ev: ev::TouchEvent| {
        let touches = ev.touches();
        match touches.length() {
            1 => {
                if let Some(t) = touches.get(0) {
                    touch.set_value(Some((t.client_x() as f64, t.client_y() as f64, 0.0)));
                    pinch.set_value(None);
                }
            }
            2 => {
                touch.set_value(None);
                pinch.set_value(pinch_state(&ev).map(|(d, cy)| (d, cy, 0.0)));
            }
            _ => {}
        }
    };

    let on_touchmove = move |ev: ev::TouchEvent| {
        let touches = ev.touches();

        if touches.length() >= 2 {
            let Some((distance, centroid_y)) = pinch_state(&ev) else {
                return;
            };
            if let Some((prev_distance, prev_y, acc)) = pinch.get_value() {
                ev.prevent_default();
                if prev_distance > 1.0 {
                    v.zoom
                        .update(|z| *z = (*z * (distance / prev_distance)).clamp(0.1, 12.0));
                }
                // Two-finger scrolling steps images, exactly as it does on a
                // trackpad.
                let acc = acc + (prev_y - centroid_y);
                let steps = (acc / TOUCH_PIXELS_PER_SLICE).trunc();
                pinch.set_value(Some((
                    distance,
                    centroid_y,
                    acc - steps * TOUCH_PIXELS_PER_SLICE,
                )));
                if steps != 0.0 {
                    v.pause_cine();
                    v.step_slice(steps as i32);
                }
            } else {
                pinch.set_value(Some((distance, centroid_y, 0.0)));
            }
            return;
        }

        let (Some((last_x, last_y, acc)), Some(t)) = (touch.get_value(), touches.get(0)) else {
            return;
        };
        ev.prevent_default();
        let (x, y) = (t.client_x() as f64, t.client_y() as f64);

        if v.is_zoomed() {
            v.pan.update(|(px, py)| {
                *px += x - last_x;
                *py += y - last_y;
            });
            touch.set_value(Some((x, y, 0.0)));
            return;
        }

        // Swiping up moves forward through the stack, matching the wheel.
        let acc = acc + (last_y - y);
        let steps = (acc / TOUCH_PIXELS_PER_SLICE).trunc();
        touch.set_value(Some((x, y, acc - steps * TOUCH_PIXELS_PER_SLICE)));
        if steps != 0.0 {
            v.pause_cine();
            v.step_slice(steps as i32);
        }
    };

    let on_touchend = move |_: ev::TouchEvent| {
        touch.set_value(None);
        pinch.set_value(None);
    };

    // Bind the gestures that need `preventDefault` directly to the element,
    // once it exists.
    create_effect(move |_| {
        let Some(stage) = stage_ref.get() else { return };
        let target: &web_sys::EventTarget = stage.as_ref();
        listen_active(target, "wheel", on_wheel);
        listen_active(target, "touchstart", on_touchstart);
        listen_active(target, "touchmove", on_touchmove);
        listen_active(target, "touchend", on_touchend);
        listen_active(target, "touchcancel", on_touchend);
    });

    let transform = move || {
        let (px, py) = v.pan.get();
        format!("translate({px}px, {py}px) scale({})", v.zoom.get())
    };

    view! {
        <div
            class="stage"
            node_ref=stage_ref
            class:zoomed=move || v.is_zoomed()
            on:mousedown=on_mousedown
            on:dblclick=move |_| v.reset_view()
            on:contextmenu=move |ev: ev::MouseEvent| ev.prevent_default()
        >
            <canvas node_ref=canvas_ref style:transform=transform></canvas>

            // A decode that failed must not take the viewer with it: say so
            // over the image and leave every other control working.
            <Show when=move || v.decode_error.get().is_some()>
                <div class="stage-message error">
                    <div class="sm-title">{move || v.t("xelray.error.title")}</div>
                    <div class="sm-body">{move || v.decode_error.get().unwrap_or_default()}</div>
                    <div class="sm-body">{move || v.t("xelray.error.hint")}</div>
                </div>
            </Show>

            <Show when=move || v.busy.get() && v.decode_error.get().is_none()>
                <div class="stage-busy">{move || v.t("xelray.loading")}</div>
            </Show>

            <Show when=move || v.overlays.get()>
                <div class="overlay tl">
                    {move || v.study.with(|s| s.as_ref().map(|st| st.info.patient_name.clone()))}
                    <br/>
                    {move || v.study.with(|s| s.as_ref().map(|st| st.info.study_date.clone()))}
                </div>
                <div class="overlay tr">
                    {move || v.study.with(|s| {
                        s.as_ref()
                            .and_then(|st| st.series.get(v.series_idx.get()))
                            .map(|se| se.label())
                    })}
                </div>
                <div class="overlay bl">
                    {move || format!("WW {:.0}  WL {:.0}", v.ww.get(), v.wl.get())}
                </div>
                <div class="overlay br">
                    {move || format!(
                        "{} / {}   ×{:.2}",
                        (v.slice_idx.get() + 1).min(v.slice_count().max(1)),
                        v.slice_count(),
                        v.zoom.get(),
                    )}
                </div>
            </Show>

            {move || v.study.with(|s| {
                s.as_ref()
                    .and_then(|st| st.series.get(v.series_idx.get()))
                    .and_then(|se| se.unsupported)
                    .map(|codec| view! {
                        <div class="warnbar">
                            {v.ta("xelray.warn.unsupported", &[("codec", codec)])}
                        </div>
                    })
            })}
        </div>
    }
}

/// Distance between the first two touch points and their midpoint's Y.
fn pinch_state(ev: &ev::TouchEvent) -> Option<(f64, f64)> {
    let touches = ev.touches();
    let (a, b) = (touches.get(0)?, touches.get(1)?);
    let dx = (a.client_x() - b.client_x()) as f64;
    let dy = (a.client_y() - b.client_y()) as f64;
    let centroid_y = (a.client_y() + b.client_y()) as f64 / 2.0;
    Some(((dx * dx + dy * dy).sqrt(), centroid_y))
}
