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

/// The cheat sheet, rendered by the `?` overlay:
/// `(group key, keys, meaning key)`.
///
/// The group and meaning are translation keys, not text — the key chips
/// themselves are the same on every keyboard and stay literal. An empty
/// group continues the one above it.
pub const HELP: &[(&str, &str, &str)] = &[
    ("xelray.help.group.images", "↑ ↓", "xelray.help.step"),
    ("", "Shift+↑ Shift+↓", "xelray.help.jump"),
    ("", "g G", "xelray.help.ends"),
    ("", "Space", "xelray.help.cine"),
    ("xelray.help.group.series", "← →", "xelray.help.series"),
    ("", "[ ]", "xelray.help.series_alt"),
    ("xelray.help.group.window", "1 2 3 4", "xelray.help.presets"),
    ("xelray.help.group.view", "= -", "xelray.help.zoom"),
    ("", "0 f", "xelray.help.fit"),
    ("xelray.help.group.layout", "s Tab", "xelray.help.panel"),
    ("", "o", "xelray.help.overlays"),
    ("", "? h", "xelray.help.this"),
    ("", "Esc", "xelray.help.close_list"),
];

/// The cheat sheet for the 3D organ view, which shares only the layout keys
/// with the slice viewer.
///
/// A separate table rather than extra rows in [`HELP`]: `0` fits the image on
/// one screen and recentres the model on the other, and a sheet that listed
/// both meanings at once would teach neither.
pub const HELP_3D: &[(&str, &str, &str)] = &[
    ("xelray.help.group.model", "1-8", "xelray.help.organs"),
    ("", "r 0", "xelray.help.recenter"),
    ("", "+ −", "xelray.help.exposure"),
    ("xelray.help.group.layout", "s Tab", "xelray.help.panel"),
    ("", "? h", "xelray.help.this"),
    ("", "Esc", "xelray.help.close_list"),
];

/// Install the window-level key handler.
pub fn install(v: Viewer) {
    // Deliberately never removed. This is installed once by the root
    // component and only ever touches signals the root owns, so it cannot
    // outlive them the way a listener registered inside a child component
    // would — those must be cleaned up, or they fire against disposed
    // signals and abort the module.
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
        // The 3D view has a key map of its own. It takes the whole screen and
        // a study is never open beside it, so the slice bindings below are not
        // merely useless here — several of them would act on a stack that does
        // not exist. Handled and returned, never fallen through.
        if let Some(bundle) = v.mesh_bundle.get_untracked() {
            match key.as_str() {
                // `r` for recentre; `0` because it is "back to the default
                // framing" in the slice viewer too.
                "r" | "0" => v.cam_reset.update(|n| *n = n.wrapping_add(1)),

                // A digit per organ, in the bundle's own order — the same
                // order the rail lists them in. Digits past the last group do
                // nothing rather than toggling a bit that renders nothing.
                "1" | "2" | "3" | "4" | "5" | "6" | "7" | "8" => {
                    let n = (key.as_bytes()[0] - b'1') as usize;
                    if n >= bundle.groups.len() {
                        return;
                    }
                    v.organ_visible.update(|m| *m ^= 1 << n);
                }

                // The slice viewer's `+`/`-` zoom; here zoom is the wheel, so
                // the pair drives brightness instead. Multiplicative steps
                // feel even across the range; the clamp keeps the picture
                // recoverable from either end.
                "+" | "=" => v.exposure.update(|e| *e = (*e * 1.15).min(2.8)),
                "-" | "_" => v.exposure.update(|e| *e = (*e / 1.15).max(0.4)),

                // Shared with the slice viewer: one rail, one signal, one key.
                "s" | "Tab" => v.rail.update(|r| *r = !*r),

                _ => return,
            }
            ev.prevent_default();
            return;
        }

        if v.study.get_untracked().is_none() {
            return;
        }

        // Space is the one navigation key that starts motion rather than
        // interrupting it.
        if key == " " || key == "Spacebar" {
            v.toggle_cine();
            ev.prevent_default();
            return;
        }

        let fast = ev.shift_key();
        let step = if fast { FAST_STEP } else { 1 };

        // Taking the wheel stops playback; it only ever resumes on Space.
        if matches!(
            key.as_str(),
            "ArrowUp" | "ArrowDown" | "PageUp" | "PageDown" | "g" | "G" | "Home" | "End"
                | "ArrowLeft" | "ArrowRight" | "[" | "]"
        ) {
            v.pause_cine();
        }

        match key.as_str() {
            // ---- images -------------------------------------------------
            "ArrowUp" | "PageUp" => v.step_slice(-step),
            "ArrowDown" | "PageDown" => v.step_slice(step),
            // `g` / `G` because Home and End need Fn on most laptops.
            "g" | "Home" => v.slice_idx.set(0),
            "G" | "End" => v.slice_idx.set(v.last_index()),

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
