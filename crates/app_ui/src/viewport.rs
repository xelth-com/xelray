//! The image stage: canvas plus the mouse and wheel gestures over it.
//!
//! Zoom and pan are a CSS transform on the canvas element rather than a
//! redraw, so panning a 512² CT costs nothing — the pixels are only
//! recomputed when the slice or the window changes.

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

#[component]
pub fn Stage(v: Viewer, canvas_ref: NodeRef<Canvas>) -> impl IntoView {
    // `(kind, last_x, last_y)` — deltas are taken against the previous move
    // so the gesture keeps working when the pointer leaves the element.
    let drag = create_rw_signal::<Option<(Drag, f64, f64)>>(None);

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
        if ev.ctrl_key() || ev.meta_key() {
            let factor = if ev.delta_y() < 0.0 { 1.15 } else { 1.0 / 1.15 };
            v.zoom.update(|z| *z = (*z * factor).clamp(0.1, 12.0));
        } else if ev.delta_y() > 0.0 {
            v.step_slice(1);
        } else if ev.delta_y() < 0.0 {
            v.step_slice(-1);
        }
    };

    let transform = move || {
        let (px, py) = v.pan.get();
        format!("translate({px}px, {py}px) scale({})", v.zoom.get())
    };

    view! {
        <div
            class="stage-canvas"
            class:panning=move || v.pan_mode.get()
            on:mousedown=on_mousedown
            on:wheel=on_wheel
            on:dblclick=move |_| v.reset_view()
            on:contextmenu=move |ev: ev::MouseEvent| ev.prevent_default()
        >
            <canvas node_ref=canvas_ref style:transform=transform></canvas>

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
        </div>
    }
}
