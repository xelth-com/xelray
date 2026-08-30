//! Translations, shared with xelth.com's backend but never dependent on it.
//!
//! The same `/i18n/{lang}` endpoint the main site uses is consulted here, so
//! a deployment under `xelth.com/M/xelray/` picks up the site's language set
//! for free. That is the *only* thing it is: a progressive enhancement.
//!
//! XelRay is open source and meant to be self-hosted, dropped on a USB stick,
//! or opened from a file:// URL by someone with a hospital CD and no network.
//! In all of those cases there is no backend, so the complete English UI is
//! compiled into the binary and rendering never waits on a fetch. The request
//! is fired after first paint, carries a short timeout, and its failure is
//! entirely unremarkable.
//!
//! Lookup order: fetched translation → embedded English → the key itself.

use std::collections::HashMap;

use leptos::*;
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

/// Languages the xelth.com backend serves. The labels are the language's own
/// name, which is the one thing every reader can recognise.
pub const SUPPORTED_LANGS: &[(&str, &str)] = &[
    ("en", "EN"),
    ("de", "DE"),
    ("ru", "РУС"),
    ("zh", "中文"),
    ("es", "ES"),
    ("ko", "한국어"),
    ("ja", "日本語"),
    ("fr", "FR"),
    ("it", "IT"),
];

/// How long to wait for the translation endpoint before giving up.
///
/// Short on purpose: the UI is already on screen in English, so a slow answer
/// is worth less than a fast failure.
const FETCH_TIMEOUT_MS: i32 = 2500;

const LANG_STORAGE_KEY: &str = "xelray.lang";

pub type Translations = HashMap<String, String>;

/// The complete UI in English, compiled in.
///
/// Every string the user can see is here; nothing falls through to a raw key
/// in a working build. Keys are namespaced `xelray.` so they can be merged
/// into xelth.com's shared table without colliding with the main site's.
pub const EN: &[(&str, &str)] = &[
    // -- landing ---------------------------------------------------------
    ("xelray.tagline", "in-browser DICOM viewer"),
    ("xelray.drop.title", "Drop DICOM files or folder here"),
    (
        "xelray.drop.sub",
        "Straight from a hospital CD — the whole DICOM folder works.",
    ),
    ("xelray.drop.folder", "Choose folder"),
    ("xelray.drop.files", "Choose files"),
    (
        "xelray.privacy",
        "🔒 Files are processed locally in your browser and never uploaded.",
    ),
    ("xelray.indexing", "{done} of {total} files indexed…"),
    ("xelray.no_dicom", "No DICOM files found in that drop."),
    (
        "xelray.none_readable",
        "None of those {total} files were readable DICOM.",
    ),
    (
        "xelray.some_ignored",
        "{skipped} of {total} files were not DICOM and were ignored.",
    ),
    // -- rail ------------------------------------------------------------
    ("xelray.rail.scans", "Scans in this study"),
    ("xelray.rail.brightness", "Brightness"),
    ("xelray.rail.size", "Size"),
    ("xelray.rail.image", "Image"),
    (
        "xelray.rail.hide_panel",
        "Hide this panel — the image fills the window (S)",
    ),
    ("xelray.rail.show_panel", "Show the controls panel (S)"),
    ("xelray.rail.language", "Language"),
    ("xelray.series.meta", "{modality} · {count} images"),
    ("xelray.series.tip", "Show this scan — {modality}, {count} images"),
    ("xelray.preset.soft", "Soft tissue"),
    ("xelray.preset.lung", "Lung"),
    ("xelray.preset.bone", "Bone"),
    ("xelray.preset.brain", "Brain"),
    (
        "xelray.preset.tip",
        "Set brightness for viewing {name} (key {key})",
    ),
    ("xelray.window.tip", "Drag on the image to fine-tune"),
    ("xelray.zoom_in", "Zoom in"),
    ("xelray.zoom_in.tip", "Make the image bigger (+)"),
    ("xelray.zoom_out", "Zoom out"),
    ("xelray.zoom_out.tip", "Make the image smaller (−)"),
    ("xelray.fit", "Fit to window"),
    (
        "xelray.fit.tip",
        "Reset zoom and position (0, or double-click the image)",
    ),
    ("xelray.hide_text", "Hide text on image"),
    ("xelray.show_text", "Show text on image"),
    (
        "xelray.text.tip",
        "Show or hide the name, date and numbers over the image (O)",
    ),
    ("xelray.scrub.tip", "Drag to move through the images"),
    ("xelray.shortcuts", "Keyboard shortcuts"),
    ("xelray.shortcuts.tip", "List every keyboard shortcut (? or H)"),
    ("xelray.open_another", "Open another study"),
    (
        "xelray.open_another.tip",
        "Close this study and choose different files",
    ),
    // -- stage -----------------------------------------------------------
    ("xelray.loading", "Loading…"),
    ("xelray.error.title", "This image could not be shown"),
    ("xelray.error.hint", "Use ↑ and ↓ to move to another image."),
    (
        "xelray.warn.unsupported",
        "Compressed with {codec}, which this build cannot read. Images cannot be shown.",
    ),
    // -- help overlay ----------------------------------------------------
    ("xelray.help.close", "Close (Esc)"),
    ("xelray.help.group.images", "Images"),
    ("xelray.help.group.series", "Series"),
    ("xelray.help.group.window", "Window"),
    ("xelray.help.group.view", "View"),
    ("xelray.help.group.layout", "Layout"),
    ("xelray.help.step", "Previous / next image"),
    ("xelray.help.jump", "Jump 10 images"),
    ("xelray.help.ends", "First / last image"),
    ("xelray.help.series", "Previous / next series"),
    ("xelray.help.series_alt", "The same, either hand"),
    ("xelray.help.presets", "Soft tissue · Lung · Bone · Brain"),
    ("xelray.help.zoom", "Zoom in / out"),
    ("xelray.help.fit", "Fit to window, undo zoom"),
    ("xelray.help.panel", "Show / hide the panel"),
    ("xelray.help.overlays", "Show / hide text over the image"),
    ("xelray.help.this", "This list"),
    ("xelray.help.close_list", "Close this list"),
    (
        "xelray.help.mouse",
        "Mouse — wheel steps images, ctrl+wheel zooms, double-click fits. \
         Drag to set brightness; once zoomed in, drag moves the image instead \
         and Shift+drag sets brightness. Middle-drag always moves.",
    ),
    (
        "xelray.help.trackpad",
        "Trackpad — two-finger scroll steps images, pinch zooms. \
         Touch — swipe to step, pinch to zoom, drag to move when zoomed.",
    ),
    (
        "xelray.help.fnkeys",
        "Home, End, PageUp and PageDown also work, if your keyboard has them \
         without holding Fn.",
    ),
];

/// Language state plus whatever the backend supplied.
#[derive(Clone, Copy)]
pub struct I18n {
    pub lang: RwSignal<String>,
    /// Empty until (and unless) a fetch succeeds.
    pub fetched: RwSignal<Translations>,
}

impl Default for I18n {
    fn default() -> Self {
        Self::new()
    }
}

impl I18n {
    pub fn new() -> Self {
        Self {
            lang: create_rw_signal(initial_lang()),
            fetched: create_rw_signal(Translations::new()),
        }
    }

    /// Translate a key.
    ///
    /// Reactive: reading `fetched` inside a view closure means the UI
    /// re-renders by itself if translations arrive later, without any of the
    /// call sites knowing that can happen.
    pub fn t(&self, key: &str) -> String {
        if let Some(hit) = self.fetched.with(|m| m.get(key).cloned()) {
            if !hit.is_empty() {
                return hit;
            }
        }
        embedded(key)
    }

    /// Translate and substitute `{name}` placeholders.
    pub fn ta(&self, key: &str, args: &[(&str, &str)]) -> String {
        let mut out = self.t(key);
        for (name, value) in args {
            out = out.replace(&format!("{{{name}}}"), value);
        }
        out
    }

    /// Switch language, remembering the choice for next time.
    pub fn set_lang(&self, lang: &str) {
        if let Some(storage) = local_storage() {
            let _ = storage.set_item(LANG_STORAGE_KEY, lang);
        }
        self.lang.set(lang.to_owned());
    }

    /// Start following the language signal. Never blocks first render: the
    /// effect runs after mount and the UI is already complete in English.
    pub fn start(&self) {
        let fetched = self.fetched;
        let lang = self.lang;
        create_effect(move |_| {
            let current = lang.get();
            if current == "en" {
                // The embedded strings *are* English; asking the server for
                // them would only risk replacing them with something older.
                fetched.set(Translations::new());
                return;
            }
            spawn_local(async move {
                if let Some(map) = fetch_translations(&current).await {
                    // A late answer for a language the user has already
                    // switched away from must not be applied.
                    if lang.get_untracked() == current {
                        fetched.set(map);
                    }
                }
            });
        });
    }
}

/// The compiled-in English string for a key, or the key itself.
fn embedded(key: &str) -> String {
    EN.iter()
        .find(|(k, _)| *k == key)
        .map(|(_, v)| (*v).to_owned())
        .unwrap_or_else(|| key.to_owned())
}

fn local_storage() -> Option<web_sys::Storage> {
    web_sys::window()?.local_storage().ok().flatten()
}

/// A remembered choice, else the browser's language, else English.
fn initial_lang() -> String {
    if let Some(saved) = local_storage().and_then(|s| s.get_item(LANG_STORAGE_KEY).ok().flatten()) {
        if is_supported(&saved) {
            return saved;
        }
    }

    let tag = web_sys::window()
        .and_then(|w| w.navigator().language())
        .unwrap_or_default()
        .to_ascii_lowercase();
    // `navigator.language` is a BCP-47 tag: `de`, `de-AT`, `zh-Hans-CN`.
    let base = tag.split(['-', '_']).next().unwrap_or("").to_owned();
    if is_supported(&base) {
        base
    } else {
        "en".to_owned()
    }
}

fn is_supported(lang: &str) -> bool {
    SUPPORTED_LANGS.iter().any(|(code, _)| *code == lang)
}

/// Ask xelth.com's translation endpoint, giving up quickly.
///
/// The URL is relative, so it resolves against whatever host the app is
/// served from — the shared backend when deployed under xelth.com, and a
/// 404 (or a connection error, or nothing at all on file://) anywhere else.
/// Every one of those outcomes is a `None` and a UI that stays English.
async fn fetch_translations(lang: &str) -> Option<Translations> {
    let window = web_sys::window()?;

    // `fetch` has no timeout of its own; an AbortController on a timer is the
    // standard way to bound it.
    let controller = web_sys::AbortController::new().ok()?;
    let options = web_sys::RequestInit::new();
    options.set_signal(Some(&controller.signal()));

    let abort = Closure::once_into_js(move || controller.abort());
    let _ = window.set_timeout_with_callback_and_timeout_and_arguments_0(
        abort.unchecked_ref(),
        FETCH_TIMEOUT_MS,
    );

    let url = format!("/i18n/{lang}");
    let response = JsFuture::from(window.fetch_with_str_and_init(&url, &options))
        .await
        .ok()?;
    let response: web_sys::Response = response.dyn_into().ok()?;
    if !response.ok() {
        return None;
    }

    let json = JsFuture::from(response.json().ok()?).await.ok()?;
    Some(to_translations(&json))
}

/// Flatten a `{key: string}` JSON object.
///
/// Done by hand rather than through serde: it is a dozen lines, it cannot
/// fail on an unexpected value type (non-strings are simply skipped), and it
/// keeps a serialisation framework out of the dependency tree.
fn to_translations(value: &JsValue) -> Translations {
    let mut map = Translations::new();
    if !value.is_object() {
        return map;
    }
    let entries = js_sys::Object::entries(&js_sys::Object::from(value.clone()));
    for entry in entries.iter() {
        let Ok(pair) = entry.dyn_into::<js_sys::Array>() else {
            continue;
        };
        let (Some(k), Some(v)) = (pair.get(0).as_string(), pair.get(1).as_string()) else {
            continue;
        };
        map.insert(k, v);
    }
    map
}
