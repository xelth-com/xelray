//! XelRay — a DICOM viewer that runs entirely in the browser tab.
//!
//! Nothing here talks to a server. Files are read with the File API, parsed
//! by [`xelray_core`] compiled to WebAssembly, and painted onto a 2D canvas.
//!
//! The layout is deliberately spartan: one collapsible rail on the left and
//! the image everywhere else. Reading a scan means looking at pixels, so
//! every band of chrome that is not the image is a band taken away from it.

mod files;
mod shortcuts;
mod viewport;

use std::collections::HashMap;
use std::rc::Rc;

use leptos::html::Canvas;
use leptos::*;
use wasm_bindgen::{Clamped, JsCast};
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement, HtmlInputElement};
use xelray_core::{Slice, Study, WINDOW_PRESETS};

const REPO_URL: &str = "https://github.com/xelth-com/xelray";

/// How far `Shift` + a step key jumps.
const FAST_STEP: i32 = 10;

/// Every piece of viewer state, bundled so the sub-views can take one prop.
///
/// Leptos signals are `Copy`, so this struct is too — it is passed by value
/// into every closure below.
#[derive(Clone, Copy)]
pub struct Viewer {
    pub study: RwSignal<Option<Rc<Study>>>,
    /// `(files parsed, files total)` while a load is in flight.
    pub progress: RwSignal<Option<(usize, usize)>>,
    pub series_idx: RwSignal<usize>,
    pub slice_idx: RwSignal<usize>,
    /// The currently displayed slice, decoded lazily by an effect.
    pub decoded: RwSignal<Option<Rc<Slice>>>,
    /// Window width and window level, in modality units (HU for CT).
    pub ww: RwSignal<f64>,
    pub wl: RwSignal<f64>,
    pub zoom: RwSignal<f64>,
    pub pan: RwSignal<(f64, f64)>,
    /// When on, a plain left-drag pans instead of adjusting window/level.
    pub pan_mode: RwSignal<bool>,
    /// Left rail visible. Off gives the image the entire viewport.
    pub rail: RwSignal<bool>,
    /// Corner text over the image.
    pub overlays: RwSignal<bool>,
    pub help: RwSignal<bool>,
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
            study: create_rw_signal(None),
            progress: create_rw_signal(None),
            series_idx: create_rw_signal(0),
            slice_idx: create_rw_signal(0),
            decoded: create_rw_signal(None),
            ww: create_rw_signal(400.0),
            wl: create_rw_signal(40.0),
            zoom: create_rw_signal(1.0),
            pan: create_rw_signal((0.0, 0.0)),
            pan_mode: create_rw_signal(false),
            rail: create_rw_signal(true),
            overlays: create_rw_signal(true),
            help: create_rw_signal(false),
            notice: create_rw_signal(None),
            drag_over: create_rw_signal(false),
        }
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

    /// Move `delta` slices, clamped to the series.
    pub fn step_slice(&self, delta: i32) {
        let count = self.slice_count();
        if count == 0 {
            return;
        }
        let next = (self.slice_idx.get_untracked() as i32 + delta).clamp(0, count as i32 - 1);
        self.slice_idx.set(next as usize);
    }

    /// Move `delta` series, clamped.
    pub fn step_series(&self, delta: i32) {
        let count = self.series_count();
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

    /// Back to fit-to-pane. The canvas is CSS-fitted, so this is just an
    /// identity transform.
    pub fn reset_view(&self) {
        self.zoom.set(1.0);
        self.pan.set((0.0, 0.0));
    }

    /// Pick a series and re-derive everything that depends on it.
    pub fn select_series(&self, idx: usize) {
        self.series_idx.set(idx);
        let count = self.slice_count();
        // Opening in the middle of the stack is far more useful than the
        // first slice, which on a CT is usually empty air.
        self.slice_idx.set(count / 2);
        self.reset_view();

        let window = self
            .study
            .with_untracked(|s| s.as_ref().and_then(|st| st.series.get(idx)?.default_window()));
        if let Some((width, center)) = window {
            self.set_window(width, center);
        }
    }

    /// Read a batch of picked or dropped files into a [`Study`].
    ///
    /// Parsing happens one file at a time with an `await` in between, so the
    /// browser keeps painting: a 1000-slice CD shows a live counter instead
    /// of a frozen tab.
    pub fn load(&self, picked: Vec<web_sys::File>) {
        let this = *self;
        spawn_local(async move {
            let picked: Vec<_> = picked
                .into_iter()
                .filter(|f| !files::looks_skippable(&f.name()))
                .collect();

            if picked.is_empty() {
                this.notice.set(Some("No DICOM files found in that drop.".into()));
                return;
            }

            this.study.set(None);
            this.decoded.set(None);
            this.notice.set(None);

            let total = picked.len();
            this.progress.set(Some((0, total)));

            let mut acc = Study::default();
            let mut by_uid: HashMap<String, usize> = HashMap::new();

            for (i, file) in picked.iter().enumerate() {
                if let Some((name, bytes)) = files::read_bytes(file).await {
                    match xelray_core::parse_instance(&name, &bytes) {
                        Ok(inst) => xelray_core::push_instance(&mut acc, &mut by_uid, inst),
                        Err(e) => acc.skipped.push((name, e.to_string())),
                    }
                }
                // Repainting the counter on every file would cost more than
                // the parsing does.
                if i % 5 == 0 || i + 1 == total {
                    this.progress.set(Some((i + 1, total)));
                }
            }

            xelray_core::finalize(&mut acc);
            this.progress.set(None);

            if acc.series.is_empty() {
                this.notice
                    .set(Some(format!("None of those {total} files were readable DICOM.")));
                return;
            }
            if !acc.skipped.is_empty() {
                this.notice.set(Some(format!(
                    "{} of {total} files were not DICOM and were ignored.",
                    acc.skipped.len()
                )));
            }

            this.study.set(Some(Rc::new(acc)));
            this.select_series(0);
        });
    }

    /// Drop the study and return to the landing screen.
    pub fn unload(&self) {
        self.study.set(None);
        self.decoded.set(None);
        self.notice.set(None);
        self.help.set(false);
        self.reset_view();
    }
}

/// Root component.
#[component]
pub fn App() -> impl IntoView {
    let v = Viewer::new();

    // Decode whenever the addressed slice changes. Decoding is kept out of
    // the load loop so that a 500-slice series costs one decode, not 500.
    create_effect(move |_| {
        let idx = v.slice_idx.get();
        let series_idx = v.series_idx.get();
        let Some(study) = v.study.get() else {
            v.decoded.set(None);
            return;
        };
        let Some(inst) = study
            .series
            .get(series_idx)
            .and_then(|s| s.instances.get(idx))
        else {
            v.decoded.set(None);
            return;
        };
        match inst.decode() {
            Ok(slice) => v.decoded.set(Some(Rc::new(slice))),
            Err(e) => {
                v.decoded.set(None);
                v.notice.set(Some(e.to_string()));
            }
        }
    });

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

    view! {
        <div class="landing">
            <div class="landing-brand">
                <span class="logo big">"XelRay"</span>
                <span class="tag">"in-browser DICOM viewer"</span>
            </div>

            <div class="dropzone">
                <div class="dz-icon">"⊕"</div>
                <h1>"Drop DICOM files or folder here"</h1>
                <p class="dz-sub">
                    "Straight from a hospital CD — the whole "
                    <code>"DICOM"</code>
                    " folder works."
                </p>

                <div class="dz-buttons">
                    <label class="btn">
                        "Choose folder"
                        // `webkitdirectory` is the only cross-browser way to
                        // pick a whole directory; Trunk leaves it verbatim.
                        <input
                            type="file"
                            webkitdirectory=""
                            directory=""
                            multiple=""
                            on:change=pick
                        />
                    </label>
                    <label class="btn ghost">
                        "Choose files"
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
                                <span>{format!("Reading {done} / {total} files…")}</span>
                            </div>
                        }
                    }}
                </Show>

                <Show when=move || v.notice.get().is_some()>
                    <p class="notice">{move || v.notice.get().unwrap_or_default()}</p>
                </Show>
            </div>

            <p class="privacy">
                "🔒 Files are processed locally in your browser and never uploaded."
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
        paint(&canvas, &slice, v.ww.get(), v.wl.get());
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
                    title="Show panel (S)"
                    on:click=move |_| v.rail.set(true)
                >"☰"</button>
            </Show>

            <viewport::Stage v canvas_ref/>
        </div>
    }
}

/// The single narrow panel that holds all the chrome.
#[component]
fn Rail(v: Viewer) -> impl IntoView {
    view! {
        <aside class="rail">
            <div class="rail-top">
                <span class="logo">"XelRay"</span>
                <button
                    class="icon"
                    title="Hide panel (S)"
                    on:click=move |_| v.rail.set(false)
                >"‹"</button>
            </div>

            <div class="rail-body">
                <div class="rail-sec">
                    <div class="rail-head">
                        "Series"
                        <span class="kbd-hint">"[ ]"</span>
                    </div>
                    {move || v.study.with(|s| {
                        let Some(st) = s.as_ref() else { return Vec::new() };
                        st.series
                            .iter()
                            .enumerate()
                            .map(|(i, se)| {
                                let label = se.label();
                                let count = se.len();
                                let modality = se.modality.clone();
                                let warnings = se.warnings.clone();
                                view! {
                                    <button
                                        class="series"
                                        class:active=move || v.series_idx.get() == i
                                        title=format!("{modality} · {count} images  (← → or [ ])")
                                        on:click=move |_| v.select_series(i)
                                    >
                                        <span class="s-label">{label}</span>
                                        <span class="s-meta">
                                            {format!("{modality} · {count}")}
                                        </span>
                                        {(!warnings.is_empty()).then(|| view! {
                                            <span class="s-warn">{warnings.join(" ")}</span>
                                        })}
                                    </button>
                                }
                            })
                            .collect::<Vec<_>>()
                    })}
                </div>

                <div class="rail-sec">
                    <div class="rail-head">
                        "Window"
                        <span class="kbd-hint">"1-4"</span>
                    </div>
                    <div class="grid2">
                        {WINDOW_PRESETS.iter().enumerate().map(|(i, (name, width, center))| {
                            let (width, center) = (*width, *center);
                            view! {
                                <button
                                    class="tool"
                                    class:active=move || {
                                        (v.ww.get() - width).abs() < 0.5
                                            && (v.wl.get() - center).abs() < 0.5
                                    }
                                    title=format!("{name} — {width:.0}/{center:.0}  (key {})", i + 1)
                                    on:click=move |_| v.set_window(width, center)
                                >{*name}</button>
                            }
                        }).collect::<Vec<_>>()}
                    </div>
                    <div class="readout">
                        {move || format!("WW {:.0}  WL {:.0}", v.ww.get(), v.wl.get())}
                    </div>
                </div>

                <div class="rail-sec">
                    <div class="rail-head">
                        "View"
                        <span class="kbd-hint">"+ - 0"</span>
                    </div>
                    <div class="grid2">
                        <button
                            class="tool"
                            class:active=move || v.pan_mode.get()
                            title="Left-drag pans instead of adjusting window/level (P)"
                            on:click=move |_| v.pan_mode.update(|p| *p = !*p)
                        >"Pan"</button>
                        <button class="tool" title="Fit to window (0 or F)"
                            on:click=move |_| v.reset_view()>"Fit"</button>
                        <button class="tool" title="Zoom in (+)"
                            on:click=move |_| v.zoom_by(1.25)>"+"</button>
                        <button class="tool" title="Zoom out (−)"
                            on:click=move |_| v.zoom_by(1.0 / 1.25)>"−"</button>
                    </div>
                    <button
                        class="tool wide"
                        class:active=move || !v.overlays.get()
                        title="Show or hide the text over the image (O)"
                        on:click=move |_| v.overlays.update(|o| *o = !*o)
                    >
                        {move || if v.overlays.get() { "Hide overlays" } else { "Show overlays" }}
                    </button>
                </div>

                <div class="rail-sec">
                    <div class="rail-head">
                        "Image"
                        <span class="kbd-hint">"↑ ↓"</span>
                    </div>
                    <div class="readout">
                        {move || {
                            let n = v.slice_count();
                            format!("{} / {}", (v.slice_idx.get() + 1).min(n.max(1)), n)
                        }}
                    </div>
                    <input
                        type="range"
                        class="scrub"
                        min="0"
                        prop:max=move || (v.slice_count().saturating_sub(1)).to_string()
                        prop:value=move || v.slice_idx.get().to_string()
                        on:input=move |ev| {
                            if let Ok(i) = event_target_value(&ev).parse::<usize>() {
                                v.slice_idx.set(i);
                            }
                        }
                    />
                </div>
            </div>

            <div class="rail-foot">
                <button
                    class="tool wide"
                    title="Keyboard shortcuts (? or H)"
                    on:click=move |_| v.help.set(true)
                >
                    "Shortcuts" <span class="kbd-hint">"?"</span>
                </button>
                <button
                    class="tool wide"
                    title="Close this study and pick another"
                    on:click=move |_| v.unload()
                >"Load another study"</button>
                <div class="rail-links">
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
                    "Keyboard shortcuts"
                    <button class="icon" on:click=move |_| v.help.set(false)>"×"</button>
                </div>
                <div class="help-grid">
                    {shortcuts::HELP.iter().map(|(group, keys, what)| view! {
                        <>
                            <div class="hk-group">{*group}</div>
                            <div class="hk-keys">
                                {keys.split(' ').map(|k| view! { <kbd>{k.replace('_', " ")}</kbd> })
                                    .collect::<Vec<_>>()}
                            </div>
                            <div class="hk-what">{*what}</div>
                        </>
                    }).collect::<Vec<_>>()}
                </div>
                <div class="help-foot">
                    <p>
                        "Mouse — wheel steps images, ctrl+wheel zooms, left-drag sets \
                         window/level, middle-drag pans, double-click fits."
                    </p>
                    <p>
                        "Trackpad — two-finger scroll steps images, pinch zooms. \
                         Touch — swipe up and down to step, pinch to zoom."
                    </p>
                    <p class="help-note">
                        "Home, End, PageUp and PageDown also work, if your keyboard \
                         has them without holding Fn."
                    </p>
                </div>
            </div>
        </div>
    }
}

/// Map modality values through the window into 8-bit grey and blit them.
///
/// The canvas is sized to the image and scaled by CSS, so this never
/// resamples: one `ImageData` per repaint, and zoom/pan cost nothing.
fn paint(canvas: &HtmlCanvasElement, slice: &Slice, ww: f64, wl: f64) {
    let (w, h) = (slice.columns as u32, slice.rows as u32);
    if w == 0 || h == 0 {
        return;
    }
    if canvas.width() != w {
        canvas.set_width(w);
    }
    if canvas.height() != h {
        canvas.set_height(h);
    }

    let ww = ww.max(1.0);
    let low = wl - ww / 2.0;
    let scale = 255.0 / ww;

    let mut rgba = vec![255u8; slice.pixels.len() * 4];
    for (px, out) in slice.pixels.iter().zip(rgba.chunks_exact_mut(4)) {
        let g = (((*px as f64) - low) * scale).clamp(0.0, 255.0) as u8;
        out[0] = g;
        out[1] = g;
        out[2] = g;
    }

    let Ok(Some(ctx)) = canvas.get_context("2d") else { return };
    let Ok(ctx) = ctx.dyn_into::<CanvasRenderingContext2d>() else {
        return;
    };
    if let Ok(image) =
        web_sys::ImageData::new_with_u8_clamped_array_and_sh(Clamped(&rgba), w, h)
    {
        let _ = ctx.put_image_data(&image, 0.0, 0.0);
    }
}
