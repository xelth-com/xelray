//! The image stage: canvas plus the pointer, wheel and touch gestures.
//!
//! Zoom and pan are a CSS transform on the canvas element rather than a
//! redraw, so panning a 512² CT costs nothing — the pixels are only
//! recomputed when the slice or the window changes. The canvas is fitted to
//! the pane by `max-width`/`max-height`, so it follows a window resize with
//! no JavaScript at all.

use leptos::html::Canvas;
use leptos::*;

use crate::Viewer;

/// What the current left/middle-button drag is doing.
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

/// Trackpad swipe distance worth one slice, for one-finger touch.
const TOUCH_PIXELS_PER_SLICE: f64 = 18.0;

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
    // `(kind, last_x, last_y)` — deltas are taken against the previous move
    // so the gesture keeps working when the pointer leaves the element.
    let drag = create_rw_signal::<Option<(Drag, f64, f64)>>(None);
    // Leftover scroll distance that has not yet added up to a whole slice.
    let scroll_acc = store_value(0.0f64);
    // One-finger touch: last Y, and accumulated swipe.
    let touch = store_value::<Option<(f64, f64)>>(None);
    // Two-finger touch: the pinch distance at the last touchmove.
    let pinch = store_value::<Option<f64>>(None);

    let _ = window_event_listener(ev::mousemove, move |ev| {
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

    let _ = window_event_listener(ev::mouseup, move |_| drag.set(None));

    let on_mousedown = move |ev: ev::MouseEvent| {
        let kind = match ev.button() {
            1 => Drag::Pan,
            0 if v.pan_mode.get_untracked() => Drag::Pan,
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
    // Deliberately minimal: one finger swipes through the stack, two fingers
    // pinch to zoom. `touch-action: none` in the CSS is what makes both
    // reach us instead of scrolling the page.

    let on_touchstart = move |ev: ev::TouchEvent| {
        let touches = ev.touches();
        match touches.length() {
            1 => {
                if let Some(t) = touches.get(0) {
                    touch.set_value(Some((t.client_y() as f64, 0.0)));
                    pinch.set_value(None);
                }
            }
            2 => {
                touch.set_value(None);
                pinch.set_value(pinch_distance(&ev));
            }
            _ => {}
        }
    };

    let on_touchmove = move |ev: ev::TouchEvent| {
        let touches = ev.touches();

        if touches.length() >= 2 {
            let Some(now) = pinch_distance(&ev) else { return };
            if let Some(prev) = pinch.get_value() {
                if prev > 1.0 {
                    ev.prevent_default();
                    v.zoom.update(|z| *z = (*z * (now / prev)).clamp(0.1, 12.0));
                }
            }
            pinch.set_value(Some(now));
            return;
        }

        let (Some((last_y, acc)), Some(t)) = (touch.get_value(), touches.get(0)) else {
            return;
        };
        ev.prevent_default();
        let y = t.client_y() as f64;
        // Swiping up moves forward through the stack, matching the wheel.
        let acc = acc + (last_y - y);
        let steps = (acc / TOUCH_PIXELS_PER_SLICE).trunc();
        touch.set_value(Some((y, acc - steps * TOUCH_PIXELS_PER_SLICE)));
        if steps != 0.0 {
            v.step_slice(steps as i32);
        }
    };

    let on_touchend = move |_: ev::TouchEvent| {
        touch.set_value(None);
        pinch.set_value(None);
    };

    let transform = move || {
        let (px, py) = v.pan.get();
        format!("translate({px}px, {py}px) scale({})", v.zoom.get())
    };

    view! {
        <div
            class="stage"
            class:panning=move || v.pan_mode.get()
            on:mousedown=on_mousedown
            on:wheel=on_wheel
            on:dblclick=move |_| v.reset_view()
            on:contextmenu=move |ev: ev::MouseEvent| ev.prevent_default()
            on:touchstart=on_touchstart
            on:touchmove=on_touchmove
            on:touchend=on_touchend
            on:touchcancel=on_touchend
        >
            <canvas node_ref=canvas_ref style:transform=transform></canvas>

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
                    .filter(|se| !se.warnings.is_empty())
                    .map(|se| view! { <div class="warnbar">{se.warnings.join(" ")}</div> })
            })}
        </div>
    }
}

/// Distance between the first two touch points, for pinch zoom.
fn pinch_distance(ev: &ev::TouchEvent) -> Option<f64> {
    let touches = ev.touches();
    let (a, b) = (touches.get(0)?, touches.get(1)?);
    let dx = (a.client_x() - b.client_x()) as f64;
    let dy = (a.client_y() - b.client_y()) as f64;
    Some((dx * dx + dy * dy).sqrt())
}
