//! The global key map.
//!
//! Bindings assume a laptop keyboard: plain arrows, letters, digits and the
//! `=`/`-` pair. `Home`, `End`, `PageUp` and `PageDown` are accepted, but
//! only ever as secondary bindings — on most laptops they sit behind an `Fn`
//! combination, so nothing is reachable *only* through them.
//!
//! The listener is on `window`, so shortcuts work without clicking the image
//! first. Typing into a form control is left alone.

use leptos::*;
use wasm_bindgen::JsCast;

use crate::{Viewer, FAST_STEP};

/// The cheat sheet, rendered by the `?` overlay: `(group, keys, meaning)`.
///
/// Keys are split on spaces into individual `<kbd>` chips.
pub const HELP: &[(&str, &str, &str)] = &[
    ("Images", "↑ ↓", "Previous / next image"),
    ("", "Shift+↑ Shift+↓", "Jump 10 images"),
    ("", "g G", "First / last image"),
    ("Series", "← →", "Previous / next series"),
    ("", "[ ]", "The same, either hand"),
    ("Window", "1 2 3 4", "Soft tissue · Lung · Bone · Brain"),
    ("View", "= -", "Zoom in / out"),
    ("", "0 f", "Fit to window"),
    ("", "p", "Pan mode — left-drag pans"),
    ("Layout", "s Tab", "Show / hide the panel"),
    ("", "o", "Show / hide text over the image"),
    ("", "? h", "This list"),
    ("", "Esc", "Close this list"),
];

/// Install the window-level key handler.
pub fn install(v: Viewer) {
    let _ = window_event_listener(ev::keydown, move |ev| {
        // Never fight the browser's own chords (copy, reload, zoom…).
        if ev.ctrl_key() || ev.meta_key() || ev.alt_key() {
            return;
        }

        let key = ev.key();

        // Escape closes whatever is open, even from inside a control.
        if key == "Escape" {
            if v.help.get_untracked() {
                v.help.set(false);
                ev.prevent_default();
            }
            return;
        }

        // Let people type. The slice scrubber is a range input, so its own
        // arrow-key behaviour takes over while it has focus — which is what
        // someone who just clicked it expects.
        if typing_target(&ev) {
            return;
        }

        // `?` works on the landing screen too; everything else needs a study.
        if key == "?" || key == "h" {
            v.help.update(|h| *h = !*h);
            ev.prevent_default();
            return;
        }
        if v.study.get_untracked().is_none() {
            return;
        }

        let fast = ev.shift_key();
        let step = if fast { FAST_STEP } else { 1 };

        match key.as_str() {
            // ---- images -------------------------------------------------
            "ArrowUp" | "PageUp" => v.step_slice(-step),
            "ArrowDown" | "PageDown" => v.step_slice(step),
            // `g` / `G` because Home and End need Fn on most laptops.
            "g" | "Home" => v.slice_idx.set(0),
            "G" | "End" => v.slice_idx.set(v.slice_count().saturating_sub(1)),

            // ---- series -------------------------------------------------
            "ArrowLeft" | "[" => v.step_series(-1),
            "ArrowRight" | "]" => v.step_series(1),

            // ---- window presets ----------------------------------------
            "1" | "2" | "3" | "4" => {
                let n = key.as_bytes()[0] - b'1';
                v.apply_preset(n as usize);
            }

            // ---- view ---------------------------------------------------
            // `+` and `_` are the shifted forms; accepting both means the
            // user never has to think about the shift key.
            "=" | "+" => v.zoom_by(1.25),
            "-" | "_" => v.zoom_by(1.0 / 1.25),
            "0" | "f" => v.reset_view(),
            "p" => v.pan_mode.update(|p| *p = !*p),

            // ---- layout -------------------------------------------------
            "s" | "Tab" => v.rail.update(|r| *r = !*r),
            "o" => v.overlays.update(|o| *o = !*o),

            _ => return,
        }

        // Only swallow keys we actually acted on, so Tab still tabs and the
        // arrows still scroll on the landing screen.
        ev.prevent_default();
    });
}

/// True when the event is headed for a text field or another control that
/// wants its own key handling.
fn typing_target(ev: &ev::KeyboardEvent) -> bool {
    let Some(target) = ev.target() else {
        return false;
    };
    let Ok(el) = target.dyn_into::<web_sys::Element>() else {
        return false;
    };
    matches!(el.tag_name().as_str(), "INPUT" | "TEXTAREA" | "SELECT")
        || el.has_attribute("contenteditable")
}
