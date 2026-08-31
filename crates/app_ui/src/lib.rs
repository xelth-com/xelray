//! XelRay — a DICOM viewer that runs entirely in the browser tab.
//!
//! Nothing here talks to a server. Files are read with the File API, parsed
//! by [`xelray_core`] compiled to WebAssembly, and painted onto a 2D canvas.
//!
//! # Why this is not just "read the files"
//!
//! A hospital CD is 500 MB and a wasm32 heap is a fraction of that, so the
//! obvious design — read everything, parse everything, keep everything —
//! dies on a real study. Instead:
//!
//! * The browser's `File` handles stay on the JS side. They are references to
//!   bytes on disk and cost nothing until read.
//! * Loading a study reads a bounded prefix of each file, keeps a few hundred
//!   bytes of metadata, and drops the rest.
//! * Pixels are decoded for one image at a time, on demand, into a
//!   byte-budgeted LRU that also holds a few neighbours for smooth scrolling.
//!
//! The result is a memory ceiling set by the cache, not by the study.

mod files;
mod i18n;
mod shortcuts;
mod viewport;

use std::collections::{HashMap, HashSet};
use std::rc::Rc;
use std::time::Duration;

use leptos::leptos_dom::helpers::IntervalHandle;

use leptos::html::Canvas;
use leptos::*;
use wasm_bindgen::{Clamped, JsCast};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, HtmlInputElement};
use xelray_core::{Slice, SliceCache, Study, HEADER_PREFIX_BYTES, WINDOW_PRESETS};

const REPO_URL: &str = "https://github.com/xelth-com/xelray";

/// How far `Shift` + a step key jumps.
const FAST_STEP: i32 = 10;

/// Steps closer together than this mean the user is scrolling, not reading,
/// and the prefetch window widens. Generous on purpose: a wider window costs
/// a few extra reads, while too narrow a one costs a visible stutter.
const FAST_STEP_MS: f64 = 250.0;

/// Cine playback rate. Fast enough to read anatomy as continuous motion,
/// slow enough that a disc-backed decode can usually keep up.
const CINE_FPS: f64 = 15.0;

/// Translation key and tint class for each of [`WINDOW_PRESETS`], in order.
///
/// The preset names live in the translation table rather than in the core
/// crate, which has no business knowing what language anyone reads.
const PRESET_KEYS: &[(&str, &str)] = &[
    ("xelray.preset.soft", "preset-soft"),
    ("xelray.preset.lung", "preset-lung"),
    ("xelray.preset.bone", "preset-bone"),
    ("xelray.preset.brain", "preset-brain"),
];

/// Every piece of viewer state, bundled so the sub-views can take one prop.
///
/// Leptos signals and `StoredValue`s are `Copy`, so this struct is too — it
/// is passed by value into every closure below.
#[derive(Clone, Copy)]
pub struct Viewer {
    /// Language state. Held here rather than in Leptos context so it is
    /// reachable from async tasks, which have no context of their own.
    pub i18n: i18n::I18n,
    pub study: RwSignal<Option<Rc<Study>>>,
    /// The browser's file handles, positionally matching
    /// [`xelray_core::Instance::file_index`]. Handles only — never contents.
    files: StoredValue<Vec<web_sys::File>>,
    /// Bounded store of decoded pixels. The memory ceiling lives here.
    cache: StoredValue<SliceCache>,
    /// Ids currently being read and decoded, so a slice is never fetched
    /// twice concurrently.
    inflight: StoredValue<HashSet<usize>>,
    /// The one id the user is waiting to see. Whichever task finishes it —
    /// the request itself or a prefetch that got there first — delivers it.
    pending: StoredValue<Option<usize>>,
    /// Bumped on every navigation. An in-flight decode whose generation is
    /// stale has been scrolled past and drops its result.
    generation: StoredValue<u64>,
    /// Reused RGBA scratch buffer. Allocating a megabyte per repaint would
    /// otherwise churn the heap on every window/level drag.
    rgba: StoredValue<Vec<u8>>,
    /// Which way the user is travelling through the stack, `+1` or `-1`.
    /// The prefetch window points this way and flips the instant it does.
    direction: StoredValue<i32>,
    /// Where we were, and when, so the next step can measure its own pace.
    last_idx: StoredValue<usize>,
    last_step_at: StoredValue<f64>,
    /// Ticker for cine playback; `None` when stopped.
    cine: StoredValue<Option<IntervalHandle>>,

    /// `(files indexed, files total)` while a load is in flight.
    pub progress: RwSignal<Option<(usize, usize)>>,
    pub series_idx: RwSignal<usize>,
    pub slice_idx: RwSignal<usize>,
    /// The currently displayed slice.
    pub decoded: RwSignal<Option<Rc<Slice>>>,
    /// A decode is in flight for the image on screen.
    pub busy: RwSignal<bool>,
    /// Why the current image could not be shown, if it could not.
    pub decode_error: RwSignal<Option<String>>,
    /// Window width and window level, in modality units (HU for CT).
    pub ww: RwSignal<f64>,
    pub wl: RwSignal<f64>,
    pub zoom: RwSignal<f64>,
    pub pan: RwSignal<(f64, f64)>,
    /// Left rail visible. Off gives the image the entire viewport.
    pub rail: RwSignal<bool>,
    /// Corner text over the image.
    pub overlays: RwSignal<bool>,
    pub help: RwSignal<bool>,
    /// Cine playback is running.
    pub playing: RwSignal<bool>,
    pub notice: RwSignal<Option<String>>,
    pub drag_over: RwSignal<bool>,
}

impl Default for Viewer {
    fn default() -> Self {
        Self::new()
    }
}

impl Viewer {
    pub fn new() -> Self {
        Self {
            i18n: i18n::I18n::new(),
            study: create_rw_signal(None),
            files: store_value(Vec::new()),
            cache: store_value(SliceCache::default()),
            inflight: store_value(HashSet::new()),
            pending: store_value(None),
            generation: store_value(0),
            rgba: store_value(Vec::new()),
            direction: store_value(1),
            last_idx: store_value(0),
            last_step_at: store_value(0.0),
            cine: store_value(None),

            progress: create_rw_signal(None),
            series_idx: create_rw_signal(0),
            slice_idx: create_rw_signal(0),
            decoded: create_rw_signal(None),
            busy: create_rw_signal(false),
            decode_error: create_rw_signal(None),
            ww: create_rw_signal(400.0),
            wl: create_rw_signal(40.0),
            zoom: create_rw_signal(1.0),
            pan: create_rw_signal((0.0, 0.0)),
            rail: create_rw_signal(true),
            overlays: create_rw_signal(true),
            help: create_rw_signal(false),
            playing: create_rw_signal(false),
            notice: create_rw_signal(None),
            drag_over: create_rw_signal(false),
        }
    }

    /// Translate a key. Reactive inside a view closure.
    pub fn t(&self, key: &'static str) -> String {
        self.i18n.t(key)
    }

    /// Translate with `{name}` substitutions.
    pub fn ta(&self, key: &'static str, args: &[(&str, &str)]) -> String {
        self.i18n.ta(key, args)
    }

    pub fn slice_count(&self) -> usize {
        self.study
            .with(|s| {
                s.as_ref()
                    .and_then(|st| st.series.get(self.series_idx.get()))
                    .map(|se| se.len())
            })
            .unwrap_or(0)
    }

    pub fn series_count(&self) -> usize {
        self.study.with(|s| s.as_ref().map_or(0, |st| st.series.len()))
    }

    /// [`Self::slice_count`] for callers that are not views.
    ///
    /// Key handlers, click handlers and the cine ticker all run outside any
    /// reactive context, where a tracked read subscribes nothing and merely
    /// earns a warning. Spelling the intent out separately keeps the tracked
    /// version honest for the views that do depend on it.
    fn slice_count_untracked(&self) -> usize {
        self.study
            .with_untracked(|s| {
                s.as_ref()
                    .and_then(|st| st.series.get(self.series_idx.get_untracked()))
                    .map(|se| se.len())
            })
            .unwrap_or(0)
    }

    /// Index of the last image, for the "jump to end" bindings.
    pub fn last_index(&self) -> usize {
        self.slice_count_untracked().saturating_sub(1)
    }

    fn series_count_untracked(&self) -> usize {
        self.study
            .with_untracked(|s| s.as_ref().map_or(0, |st| st.series.len()))
    }

    /// True once the image is magnified past its fitted size, which is when a
    /// left-drag becomes a pan instead of a window/level adjustment.
    pub fn is_zoomed(&self) -> bool {
        self.zoom.get() > 1.0001
    }

    /// Move `delta` slices, clamped to the series.
    pub fn step_slice(&self, delta: i32) {
        let count = self.slice_count_untracked();
        if count == 0 {
            return;
        }
        let next = (self.slice_idx.get_untracked() as i32 + delta).clamp(0, count as i32 - 1);
        self.slice_idx.set(next as usize);
    }

    /// Move `delta` series, clamped.
    pub fn step_series(&self, delta: i32) {
        let count = self.series_count_untracked();
        if count == 0 {
            return;
        }
        let next = (self.series_idx.get_untracked() as i32 + delta).clamp(0, count as i32 - 1);
        if next as usize != self.series_idx.get_untracked() {
            self.select_series(next as usize);
        }
    }

    pub fn set_window(&self, width: f64, center: f64) {
        self.ww.set(width.max(1.0));
        self.wl.set(center);
    }

    /// Apply preset `n` (0-based), if it exists.
    pub fn apply_preset(&self, n: usize) {
        if let Some((_, width, center)) = WINDOW_PRESETS.get(n) {
            self.set_window(*width, *center);
        }
    }

    pub fn zoom_by(&self, factor: f64) {
        self.zoom.update(|z| *z = (*z * factor).clamp(0.1, 12.0));
    }

    /// Back to fitted size and centred. The canvas is CSS-fitted, so this is
    /// just an identity transform.
    pub fn reset_view(&self) {
        self.zoom.set(1.0);
        self.pan.set((0.0, 0.0));
    }

    /// Pick a series and re-derive everything that depends on it.
    pub fn select_series(&self, idx: usize) {
        self.stop_cine();
        self.series_idx.set(idx);
        let count = self.slice_count_untracked();
        // Opening in the middle of the stack is far more useful than the
        // first slice, which on a CT is usually empty air.
        self.slice_idx.set(count / 2);
        self.reset_view();

        let window = self
            .study
            .with_untracked(|s| s.as_ref().and_then(|st| st.series.get(idx)?.default_window));
        if let Some((width, center)) = window {
            self.set_window(width, center);
        }
    }

    // ---- cache plumbing --------------------------------------------------
    // `StoredValue::update_value` hands out `&mut` but returns nothing, so
    // reads that mutate LRU order have to smuggle their result out.

    fn cache_get(&self, id: usize) -> Option<Rc<Slice>> {
        let mut out = None;
        self.cache.update_value(|c| out = c.get(id));
        out
    }

    fn claim_inflight(&self, id: usize) -> bool {
        let mut claimed = false;
        self.inflight.update_value(|s| claimed = s.insert(id));
        claimed
    }

    fn release_inflight(&self, id: usize) {
        self.inflight.update_value(|s| {
            s.remove(&id);
        });
    }

    /// `(cache id, file index)` for an image in the current series.
    fn locate(&self, slice_idx: usize) -> Option<(usize, usize)> {
        self.study.with_untracked(|s| {
            let inst = s
                .as_ref()?
                .series
                .get(self.series_idx.get_untracked())?
                .instances
                .get(slice_idx)?;
            Some((inst.id, inst.file_index))
        })
    }

    /// Index a batch of picked or dropped files.
    ///
    /// Only a prefix of each file is read, and only its metadata is kept, so
    /// the cost of this pass is set by the number of images rather than by
    /// their size. The `await` on every file is also what keeps the tab
    /// responsive: a thousand images stream in behind a counter.
    pub fn load(&self, picked: Vec<web_sys::File>) {
        let this = *self;
        spawn_local(async move {
            let picked: Vec<_> = picked
                .into_iter()
                .filter(|f| !files::looks_skippable(&f.name()))
                .collect();

            if picked.is_empty() {
                this.notice.set(Some(this.t("xelray.no_dicom")));
                return;
            }

            this.reset_study();
            let total = picked.len();
            this.progress.set(Some((0, total)));

            let mut acc = Study::default();
            let mut by_uid: HashMap<String, usize> = HashMap::new();

            for (i, file) in picked.iter().enumerate() {
                let name = file.name();

                // The prefix is enough for essentially every real image; the
                // full read is a fallback for the rare header that runs past
                // it, and its bytes are dropped as soon as it is parsed.
                let parsed = match files::read_prefix(file, HEADER_PREFIX_BYTES).await {
                    Some(prefix) => match xelray_core::parse_header(i, &name, &prefix) {
                        Err(e) if e.is_incomplete() => match files::read_all(file).await {
                            Some(all) => xelray_core::parse_header(i, &name, &all),
                            None => Err(e),
                        },
                        other => other,
                    },
                    None => {
                        acc.skipped.push((name, "could not be read".into()));
                        continue;
                    }
                };

                match parsed {
                    Ok(h) => xelray_core::push_header(&mut acc, &mut by_uid, h),
                    Err(e) => acc.skipped.push((name, e.to_string())),
                }

                // Repainting the counter on every file would cost more than
                // the indexing does.
                if i % 5 == 0 || i + 1 == total {
                    this.progress.set(Some((i + 1, total)));
                }
            }

            xelray_core::finalize(&mut acc);
            this.progress.set(None);

            if acc.series.is_empty() {
                this.notice.set(Some(
                    this.ta("xelray.none_readable", &[("total", &total.to_string())]),
                ));
                return;
            }
            if !acc.skipped.is_empty() {
                this.notice.set(Some(this.ta(
                    "xelray.some_ignored",
                    &[
                        ("skipped", &acc.skipped.len().to_string()),
                        ("total", &total.to_string()),
                    ],
                )));
            }

            this.files.set_value(picked);
            this.study.set(Some(Rc::new(acc)));
            this.select_series(0);
        });
    }

    /// Show the image the signals currently point at, decoding if needed.
    ///
    /// A cache hit is synchronous, which is the common case while scrolling
    /// thanks to the prefetch below. A miss leaves the previous image on
    /// screen rather than blanking it, and swaps only when the new one is
    /// ready.
    fn show_current(&self) {
        let this = *self;
        let here = self.slice_idx.get();
        let fast = self.track_motion(here);
        let Some((id, file_index)) = self.locate(here) else {
            self.decoded.set(None);
            self.pending.set_value(None);
            return;
        };

        // Every navigation invalidates whatever prefetching was in flight.
        let generation = self.generation.get_value().wrapping_add(1);
        self.generation.set_value(generation);

        // Record what is wanted *before* starting any work, so a decode
        // already running for this id knows to deliver it on the way out.
        self.pending.set_value(Some(id));

        if let Some(slice) = self.cache_get(id) {
            self.decoded.set(Some(slice));
            self.decode_error.set(None);
            self.busy.set(false);
            self.pending.set_value(None);
        } else {
            self.busy.set(true);
            // `None` generation: a request the user is waiting on is never
            // abandoned, however much scrolling happens around it.
            fetch(this, id, file_index, None);
        }

        // Warm what is about to be looked at. The window points the way the
        // user is travelling and is issued nearest-first, so the images most
        // likely to be needed win the race for the disc.
        let count = self.slice_count();
        for n in xelray_core::prefetch_order(here, count, self.direction.get_value(), fast) {
            if let Some((nid, nfi)) = self.locate(n) {
                if !self.cache.with_value(|c| c.contains(nid)) {
                    fetch(this, nid, nfi, Some(generation));
                }
            }
        }
    }

    /// Note where we are and how fast we got here.
    ///
    /// Returns whether the user is moving quickly enough to want a wider
    /// prefetch window. Cine always counts as moving.
    fn track_motion(&self, idx: usize) -> bool {
        let previous = self.last_idx.get_value();
        let now = js_sys::Date::now();
        let mut fast = self.playing.get_untracked();

        if idx != previous {
            // Flip the window on the very step that reverses, not the one
            // after — the whole point is not to spend a beat pointing the
            // wrong way.
            self.direction
                .set_value(if idx > previous { 1 } else { -1 });
            fast |= now - self.last_step_at.get_value() < FAST_STEP_MS;
            self.last_step_at.set_value(now);
            self.last_idx.set_value(idx);
        }
        fast
    }

    // ---- cine ------------------------------------------------------------

    /// Start or stop automatic playback.
    pub fn toggle_cine(&self) {
        if self.playing.get_untracked() {
            self.stop_cine();
        } else {
            self.start_cine();
        }
    }

    fn start_cine(&self) {
        if self.slice_count_untracked() < 2 {
            return;
        }
        self.stop_cine();
        let this = *self;
        let handle = set_interval_with_handle(
            move || this.cine_tick(),
            Duration::from_millis((1000.0 / CINE_FPS) as u64),
        )
        .ok();
        self.cine.set_value(handle);
        self.playing.set(true);
    }

    pub fn stop_cine(&self) {
        if let Some(handle) = self.cine.get_value() {
            handle.clear();
        }
        self.cine.set_value(None);
        self.playing.set(false);
    }

    /// Stop playback because the user took the wheel.
    ///
    /// Every manual navigation calls this; playback only ever resumes by an
    /// explicit toggle.
    pub fn pause_cine(&self) {
        if self.playing.get_untracked() {
            self.stop_cine();
        }
    }

    /// One frame of playback.
    ///
    /// Advances regardless of whether the last image finished decoding.
    /// Frames are dropped rather than queued: each tick supersedes the last
    /// request, and the generation counter makes the abandoned one discard
    /// its result, so a slow disc makes playback skip rather than fall
    /// progressively further behind.
    fn cine_tick(&self) {
        let count = self.slice_count_untracked();
        if count < 2 {
            self.stop_cine();
            return;
        }
        // Wrap: a stack read end to end reads best as a loop.
        self.slice_idx
            .set((self.slice_idx.get_untracked() + 1) % count);
    }

    /// Hand a freshly cached slice to the screen, if it is the one being
    /// waited for.
    fn deliver(&self, id: usize) {
        if self.pending.get_value() != Some(id) {
            return;
        }
        if let Some(slice) = self.cache_get(id) {
            self.decoded.set(Some(slice));
            self.decode_error.set(None);
            self.busy.set(false);
            self.pending.set_value(None);
        }
    }

    /// Drop the study and everything derived from it.
    fn reset_study(&self) {
        self.direction.set_value(1);
        self.last_idx.set_value(0);
        self.last_step_at.set_value(0.0);
        self.study.set(None);
        self.decoded.set(None);
        self.decode_error.set(None);
        self.notice.set(None);
        self.busy.set(false);
        self.files.set_value(Vec::new());
        self.cache.update_value(|c| c.clear());
        self.inflight.update_value(|s| s.clear());
        self.pending.set_value(None);
        self.generation.update_value(|g| *g = g.wrapping_add(1));
    }

    /// Drop the study and return to the landing screen.
    pub fn unload(&self) {
        self.stop_cine();
        self.reset_study();
        self.help.set(false);
        self.reset_view();
    }
}

/// Read one file and decode it into the cache.
///
/// `prefetch_generation` is `Some` for speculative work, which is abandoned
/// as soon as the user scrolls past it — reading half a megabyte off disk for
/// an image already gone by is the waste worth avoiding. A request the user
/// is actually waiting on passes `None` and always runs to completion.
///
/// Whoever finishes calls [`Viewer::deliver`], so a display request that
/// arrives while a prefetch for the same image is already in flight is served
/// by that prefetch rather than starting a second read.
fn fetch(v: Viewer, id: usize, file_index: usize, prefetch_generation: Option<u64>) {
    // Someone is already fetching this; their `deliver` will cover us.
    if !v.claim_inflight(id) {
        return;
    }

    spawn_local(async move {
        // Abandoning speculative work is only safe while nobody is waiting
        // on it — otherwise the abort would strand the request that just
        // adopted this in-flight fetch, and the image would never appear.
        macro_rules! bail_if_stale {
            () => {
                if let Some(generation) = prefetch_generation {
                    if v.generation.get_value() != generation {
                        v.release_inflight(id);
                        if v.pending.get_value() == Some(id) {
                            // Someone started waiting for it meanwhile.
                            // Restart as a real request, which cannot bail.
                            fetch(v, id, file_index, None);
                        }
                        return;
                    }
                }
            };
        }

        bail_if_stale!();

        let file = v.files.with_value(|f| f.get(file_index).cloned());
        let bytes = match file {
            Some(f) => files::read_all(&f).await,
            None => None,
        };

        // Reading was async; check again before paying for the decode.
        bail_if_stale!();

        let result = match bytes {
            Some(b) => xelray_core::decode_slice(&b),
            None => Err(xelray_core::XelRayError::Decode(
                "the file could not be read — has it been moved or ejected?".into(),
            )),
        };
        v.release_inflight(id);

        match result {
            Ok(slice) => {
                v.cache.update_value(|c| c.insert(id, Rc::new(slice)));
                v.deliver(id);
            }
            Err(e) => {
                // Fail soft: report it on the image and leave every other
                // control working. A failure nobody is waiting for stays
                // silent — the user has not asked for that image yet.
                if v.pending.get_value() == Some(id) {
                    v.decode_error.set(Some(e.to_string()));
                    v.decoded.set(None);
                    v.busy.set(false);
                    v.pending.set_value(None);
                }
            }
        }
    });
}

/// Root component.
#[component]
pub fn App() -> impl IntoView {
    let v = Viewer::new();

    // Re-resolve the displayed image whenever the address of it changes.
    create_effect(move |_| {
        v.slice_idx.track();
        v.series_idx.track();
        v.study.track();
        v.show_current();
    });

    v.i18n.start();
    shortcuts::install(v);

    let on_drop = move |ev: ev::DragEvent| {
        ev.prevent_default();
        v.drag_over.set(false);
        let Some(dt) = ev.data_transfer() else { return };
        spawn_local(async move {
            let picked = files::from_data_transfer(&dt).await;
            v.load(picked);
        });
    };

    view! {
        <div
            class="app"
            class:dropping=move || v.drag_over.get()
            on:dragover=move |ev: ev::DragEvent| {
                ev.prevent_default();
                v.drag_over.set(true);
            }
            // `dragleave` also fires when the pointer crosses between child
            // elements; only a leave with no new target means the drag has
            // actually left the window, so the highlight stops flickering.
            on:dragleave=move |ev: ev::DragEvent| {
                if ev.related_target().is_none() {
                    v.drag_over.set(false);
                }
            }
            on:drop=on_drop
        >
            <Show
                when=move || v.study.get().is_some()
                fallback=move || view! { <Landing v/> }
            >
                <ViewerPane v/>
            </Show>

            <Show when=move || v.help.get()>
                <HelpOverlay v/>
            </Show>
        </div>
    }
}

/// The empty state: drop zone, picker buttons, privacy note.
///
/// This is the only screen that shows the wordmark large; once a study is
/// open, branding shrinks into the rail.
#[component]
fn Landing(v: Viewer) -> impl IntoView {
    let pick = move |ev: ev::Event| {
        let input: HtmlInputElement = event_target(&ev);
        if let Some(list) = input.files() {
            v.load(files::from_file_list(&list));
        }
    };

    // Chromium can open a folder without claiming anything is being
    // uploaded; everyone else gets the `webkitdirectory` input. Checked once
    // per mount, by feature detection rather than by user agent.
    let native_folder_picker = files::has_directory_picker();

    let choose_folder = move |_| {
        // Opened synchronously inside the click handler: the browser only
        // allows the picker while the user's activation is still live, and
        // awaiting first would spend it.
        let promise = files::open_directory_picker();
        spawn_local(async move {
            // `None` is a cancelled dialog — the expected outcome half the
            // time, and nothing to report.
            if let Some(picked) = files::awaited_picker_files(promise).await {
                v.load(picked);
            }
        });
    };

    view! {
        <div class="landing">
            <div class="landing-brand">
                <span class="logo big">"XelRay"</span>
                <span class="tag">{move || v.t("xelray.tagline")}</span>
            </div>

            <div class="dropzone">
                <div class="dz-icon">"⊕"</div>
                <h1>{move || v.t("xelray.drop.title")}</h1>
                <p class="dz-sub">{move || v.t("xelray.drop.sub")}</p>

                <div class="dz-buttons">
                    {if native_folder_picker {
                        view! {
                            <button class="btn" on:click=choose_folder>
                                {move || v.t("xelray.drop.folder")}
                            </button>
                        }
                        .into_view()
                    } else {
                        view! {
                            <label class="btn">
                                {move || v.t("xelray.drop.folder")}
                                // The only way to pick a directory in Firefox
                                // and Safari. Trunk leaves the attribute
                                // verbatim.
                                <input
                                    type="file"
                                    webkitdirectory=""
                                    directory=""
                                    multiple=""
                                    on:change=pick
                                />
                            </label>
                        }
                        .into_view()
                    }}
                    <label class="btn ghost">
                        {move || v.t("xelray.drop.files")}
                        <input type="file" multiple="" on:change=pick />
                    </label>
                </div>

                <Show when=move || v.progress.get().is_some()>
                    {move || {
                        let (done, total) = v.progress.get().unwrap_or((0, 0));
                        let pct = if total == 0 { 0.0 } else { done as f64 * 100.0 / total as f64 };
                        view! {
                            <div class="progress">
                                <div class="bar"><div class="fill" style:width=format!("{pct:.1}%")></div></div>
                                <span>{v.ta(
                                    "xelray.indexing",
                                    &[("done", &done.to_string()), ("total", &total.to_string())],
                                )}</span>
                            </div>
                        }
                    }}
                </Show>

                <Show when=move || v.notice.get().is_some()>
                    <p class="notice">{move || v.notice.get().unwrap_or_default()}</p>
                </Show>
            </div>

            <p class="privacy">
                <span class="lock" aria-hidden="true">"🔒"</span>
                <span>{move || v.t("xelray.privacy")}</span>
            </p>

            <p class="landing-links">
                <a href="https://xelth.com">"xelth.com"</a>
                " · "
                <a href=REPO_URL target="_blank" rel="noreferrer">"GitHub"</a>
                " · MIT"
            </p>
        </div>
    }
}

/// Rail + stage. No header, no footer — the image gets everything else.
#[component]
fn ViewerPane(v: Viewer) -> impl IntoView {
    let canvas_ref = create_node_ref::<Canvas>();

    // Repaint on any change to the image or the window.
    create_effect(move |_| {
        let (Some(slice), Some(canvas)) = (v.decoded.get(), canvas_ref.get()) else {
            return;
        };
        let (ww, wl) = (v.ww.get(), v.wl.get());
        v.rgba.update_value(|buf| paint(&canvas, &slice, ww, wl, buf));
    });

    view! {
        <div class="viewer">
            <Show when=move || v.rail.get()>
                <Rail v/>
            </Show>

            // When the rail is hidden, one small floating control brings it
            // back — discoverable for the mouse, `S` for the keyboard.
            <Show when=move || !v.rail.get()>
                <button
                    class="rail-tab"
                    title=move || v.t("xelray.rail.show_panel")
                    on:click=move |_| v.rail.set(true)
                >"☰"</button>
            </Show>

            <viewport::Stage v canvas_ref/>
        </div>
    }
}

/// The single narrow panel that holds all the chrome.
///
/// Every label says what will happen, not what mode it names — the audience
/// is someone holding a CD from a hospital, not a radiographer.
#[component]
fn Rail(v: Viewer) -> impl IntoView {
    view! {
        <aside class="rail">
            <div class="rail-top">
                <span class="logo">"XelRay"</span>
                <button
                    class="icon"
                    title=move || v.t("xelray.rail.hide_panel")
                    on:click=move |_| v.rail.set(false)
                >"‹"</button>
            </div>

            <div class="rail-body">
                <div class="rail-sec">
                    <div class="rail-head nav">
                        {move || v.t("xelray.rail.scans")}
                        <span class="kbd-hint">"[ ]"</span>
                    </div>
                    {move || v.study.with(|s| {
                        let Some(st) = s.as_ref() else { return Vec::new() };
                        st.series
                            .iter()
                            .enumerate()
                            .map(|(i, se)| {
                                let label = se.label();
                                let count = se.len().to_string();
                                let modality = se.modality.clone();
                                let args = [
                                    ("modality", modality.as_str()),
                                    ("count", count.as_str()),
                                ];
                                let unsupported = se.unsupported;
                                view! {
                                    <button
                                        class="series"
                                        class:active=move || v.series_idx.get() == i
                                        title=v.ta("xelray.series.tip", &args)
                                        on:click=move |_| v.select_series(i)
                                    >
                                        <span class="s-label">{label}</span>
                                        <span class="s-meta">
                                            {v.ta("xelray.series.meta", &args)}
                                        </span>
                                        {unsupported.map(|codec| view! {
                                            <span class="s-warn">
                                                {v.ta("xelray.warn.unsupported",
                                                      &[("codec", codec)])}
                                            </span>
                                        })}
                                    </button>
                                }
                            })
                            .collect::<Vec<_>>()
                    })}
                </div>

                <div class="rail-sec">
                    <div class="rail-head window">
                        {move || v.t("xelray.rail.brightness")}
                        <span class="kbd-hint">"1-4"</span>
                    </div>
                    <div class="grid2">
                        {PRESET_KEYS.iter().enumerate().map(|(i, (key, tint))| {
                            let (_, width, center) = WINDOW_PRESETS[i];
                            view! {
                                <button
                                    class=format!("tool chip preset {tint}")
                                    class:active=move || {
                                        (v.ww.get() - width).abs() < 0.5
                                            && (v.wl.get() - center).abs() < 0.5
                                    }
                                    title=move || v.ta("xelray.preset.tip", &[
                                        ("name", &v.t(key).to_lowercase()),
                                        ("key", &(i + 1).to_string()),
                                    ])
                                    on:click=move |_| v.set_window(width, center)
                                >{move || v.t(key)}</button>
                            }
                        }).collect::<Vec<_>>()}
                    </div>
                    <div class="readout" title=move || v.t("xelray.window.tip")>
                        {move || format!("WW {:.0}  WL {:.0}", v.ww.get(), v.wl.get())}
                    </div>
                </div>

                <div class="rail-sec">
                    <div class="rail-head view">
                        {move || v.t("xelray.rail.size")}
                        <span class="kbd-hint">"+ - 0"</span>
                    </div>
                    <div class="grid2">
                        <button class="tool chip zoom-in" title=move || v.t("xelray.zoom_in.tip")
                            on:click=move |_| v.zoom_by(1.25)
                        >{move || v.t("xelray.zoom_in")}</button>
                        <button class="tool chip zoom-out" title=move || v.t("xelray.zoom_out.tip")
                            on:click=move |_| v.zoom_by(1.0 / 1.25)
                        >{move || v.t("xelray.zoom_out")}</button>
                    </div>
                    <button
                        class="tool wide chip fit"
                        title=move || v.t("xelray.fit.tip")
                        on:click=move |_| v.reset_view()
                    >{move || v.t("xelray.fit")}</button>
                    <button
                        class="tool wide"
                        title=move || v.t("xelray.text.tip")
                        on:click=move |_| v.overlays.update(|o| *o = !*o)
                    >
                        {move || if v.overlays.get() {
                            v.t("xelray.hide_text")
                        } else {
                            v.t("xelray.show_text")
                        }}
                    </button>
                </div>

                <div class="rail-sec">
                    <div class="rail-head nav">
                        {move || v.t("xelray.rail.image")}
                        <span class="kbd-hint">"↑ ↓"</span>
                    </div>
                    <div class="readout">
                        {move || {
                            let n = v.slice_count();
                            format!("{} / {}", (v.slice_idx.get() + 1).min(n.max(1)), n)
                        }}
                    </div>
                    <button
                        class="tool wide chip cine"
                        class:active=move || v.playing.get()
                        title=move || v.t("xelray.cine.tip")
                        on:click=move |_| v.toggle_cine()
                    >
                        {move || if v.playing.get() {
                            v.t("xelray.cine.pause")
                        } else {
                            v.t("xelray.cine.play")
                        }}
                        <span class="kbd-hint">"Space"</span>
                    </button>
                    <input
                        type="range"
                        class="scrub"
                        title=move || v.t("xelray.scrub.tip")
                        min="0"
                        prop:max=move || (v.slice_count().saturating_sub(1)).to_string()
                        prop:value=move || v.slice_idx.get().to_string()
                        on:input=move |ev| {
                            if let Ok(i) = event_target_value(&ev).parse::<usize>() {
                                v.pause_cine();
                                v.slice_idx.set(i);
                            }
                        }
                    />
                </div>
            </div>

            <div class="rail-foot">
                <button
                    class="tool wide"
                    title=move || v.t("xelray.shortcuts.tip")
                    on:click=move |_| v.help.set(true)
                >
                    {move || v.t("xelray.shortcuts")}
                    <span class="kbd-hint">"?"</span>
                </button>
                <button
                    class="tool wide"
                    title=move || v.t("xelray.open_another.tip")
                    on:click=move |_| v.unload()
                >{move || v.t("xelray.open_another")}</button>

                <div class="rail-links">
                    // Unobtrusive by design: a bare select that looks like the
                    // link row it sits in, not a control competing for
                    // attention next to the image.
                    <select
                        class="lang"
                        title=move || v.t("xelray.rail.language")
                        prop:value=move || v.i18n.lang.get()
                        on:change=move |ev| v.i18n.set_lang(&event_target_value(&ev))
                    >
                        {i18n::SUPPORTED_LANGS.iter().map(|(code, label)| view! {
                            <option value=*code>{*label}</option>
                        }).collect::<Vec<_>>()}
                    </select>
                    " · "
                    <a href="https://xelth.com">"xelth.com"</a>
                    " · "
                    <a href=REPO_URL target="_blank" rel="noreferrer">"GitHub"</a>
                </div>
            </div>
        </aside>
    }
}

/// The `?` cheat sheet.
#[component]
fn HelpOverlay(v: Viewer) -> impl IntoView {
    view! {
        <div class="help-backdrop" on:click=move |_| v.help.set(false)>
            // Clicks inside the card must not fall through to the backdrop.
            <div class="help-card" on:click=move |ev: ev::MouseEvent| ev.stop_propagation()>
                <div class="help-title">
                    {move || v.t("xelray.shortcuts")}
                    <button class="icon" title=move || v.t("xelray.help.close")
                        on:click=move |_| v.help.set(false)>"×"</button>
                </div>
                <div class="help-grid">
                    {shortcuts::HELP.iter().map(|(group, keys, what)| view! {
                        <>
                            // An empty group key continues the row above it.
                            // `xelray.help.group.view` -> class `view`, so the
                            // cheat sheet is colour-coded the same way the
                            // rail is.
                            <div class=format!(
                                "hk-group {}",
                                group.rsplit('.').next().unwrap_or(""),
                            )>
                                {move || if group.is_empty() {
                                    String::new()
                                } else {
                                    v.t(group)
                                }}
                            </div>
                            <div class="hk-keys">
                                {keys.split(' ').map(|k| view! { <kbd>{k.replace('_', " ")}</kbd> })
                                    .collect::<Vec<_>>()}
                            </div>
                            <div class="hk-what">{move || v.t(what)}</div>
                        </>
                    }).collect::<Vec<_>>()}
                </div>
                <div class="help-foot">
                    <p>{move || v.t("xelray.help.mouse")}</p>
                    <p>{move || v.t("xelray.help.trackpad")}</p>
                    <p class="help-note">{move || v.t("xelray.help.fnkeys")}</p>
                </div>
            </div>
        </div>
    }
}

/// Map modality values through the window into 8-bit grey and blit them.
///
/// `buf` is reused between calls: a window/level drag repaints on every mouse
/// move, and allocating a megabyte each time would churn the heap for nothing.
/// The canvas is sized to the image and scaled by CSS, so this never
/// resamples — zoom and pan cost no repaint at all.
fn paint(canvas: &HtmlCanvasElement, slice: &Slice, ww: f64, wl: f64, buf: &mut Vec<u8>) {
    let (w, h) = (slice.columns as u32, slice.rows as u32);
    if w == 0 || h == 0 {
        return;
    }
    let needed = slice.rows * slice.columns * 4;
    if slice.pixels.len() < slice.rows * slice.columns {
        return;
    }

    if canvas.width() != w {
        canvas.set_width(w);
    }
    if canvas.height() != h {
        canvas.set_height(h);
    }

    if buf.len() != needed {
        buf.clear();
        buf.resize(needed, 255);
    }

    let ww = ww.max(1.0);
    let low = wl - ww / 2.0;
    let scale = 255.0 / ww;

    for (px, out) in slice.pixels.iter().zip(buf.chunks_exact_mut(4)) {
        let g = (((*px as f64) - low) * scale).clamp(0.0, 255.0) as u8;
        out[0] = g;
        out[1] = g;
        out[2] = g;
        out[3] = 255;
    }

    let Ok(Some(ctx)) = canvas.get_context("2d") else { return };
    let Ok(ctx) = ctx.dyn_into::<CanvasRenderingContext2d>() else {
        return;
    };
    if let Ok(image) = web_sys::ImageData::new_with_u8_clamped_array_and_sh(Clamped(buf), w, h) {
        let _ = ctx.put_image_data(&image, 0.0, 0.0);
    }
}
