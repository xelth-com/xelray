"""Build a single self-contained interactive 3D HTML from the segmentation.

Reads the label map produced by `segment.py`, extracts one surface mesh per organ
group with marching cubes, and writes `out/render3d.html` -- a standalone page
(plotly.js embedded) with orbit/zoom, per-organ visibility toggles, and a
language picker covering all nine XelRay UI languages.

The render gets the whole viewport. All of the chrome -- caption, organ list,
picker, hint, credit -- sits in one narrow left rail styled like the XelRay
DICOM viewer's, and that rail folds away entirely (chevron, or `s` / `Tab`).

All computation and rendering is offline: the mesh data and plotly itself are in
the file, and English is embedded, so the page is fully usable with no network.
The only thing fetched at view time is a translation bundle for a *non-English*
language, from `https://xelth.com/i18n/{lang}`; it is cached in localStorage, so
after the first online view the reader's chosen language also works offline.

    python render3d.py --out out

Organ meshes come from the 3 mm model-space label map by default: it is ~40x fewer
voxels than the original grid and, after a light gaussian pre-smooth, the surfaces
are visually indistinguishable at this zoom level. `--space orig` uses the
full-resolution map instead (slower, much larger HTML).

The tumour estimate is derived separately at full resolution -- see `tumour_mask`.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import SimpleITK as sitk
from scipy import ndimage as ndi
from skimage import measure

LANGS = ["en", "de", "ru", "es", "fr", "it", "ja", "ko", "zh"]

# Native language names for the picker.
LANG_NAMES = {
    "en": "English", "de": "Deutsch", "ru": "Русский", "es": "Español",
    "fr": "Français", "it": "Italiano", "ja": "日本語", "ko": "한국어", "zh": "中文",
}

# Only English is embedded, so the page renders fully offline out of the box.
# The other eight languages are fetched at view time from the XelRay i18n endpoint
# and then cached in localStorage, so a later *offline* reopen still shows the
# language the reader last picked. Keys mirror `xelray.organ.*` on the server;
# the modebar tooltips below mirror `xelray.viewer3d.*`.
EN = {
    "kidney_left": "Left kidney",
    "kidney_right": "Right kidney",
    "liver": "Liver",
    "spleen": "Spleen",
    "pancreas": "Pancreas",
    "aorta": "Aorta",
    "bones": "Bones",
    "tumor_region": "Tumor region (estimate)",
    "ai_disclaimer": "AI segmentation, not a diagnosis.",
    "unit_ml": "ml",
}

I18N_URL = "https://xelth.com/i18n/"
I18N_PREFIX = "xelray.organ."
I18N_PREFIX_UI = "xelray.viewer3d."

# Plotly's own modebar tooltips. A plotly locale dictionary is keyed by plotly's
# *English* strings, so the English value of each key is also the dictionary key
# it translates -- which is why this map doubles as the English fallback. The
# strings must match the bundle exactly (note the upper-case "PNG").
MODEBAR = {
    "download_png": "Download plot as a PNG",
    "zoom": "Zoom",
    "pan": "Pan",
    "orbital": "Orbital rotation",
    "turntable": "Turntable rotation",
    "reset_camera": "Reset camera to default",
    "reset_camera_saved": "Reset camera to last save",
    "hover_toggle": "Toggle show closest data on hover",
}

PLOT_CONFIG = {
    "displaylogo": False,
    "responsive": True,
    # A medical document must never offer to post its data to a third-party
    # cloud; this removes plotly's "Share chart…" (chart-studio) button. The
    # button is `sendChartToCloud` in plotly 5+ -- `sendDataToCloud` is the
    # old name, kept here so the removal holds either way.
    "modeBarButtonsToRemove": ["sendDataToCloud", "sendChartToCloud"],
}

# The document title is language-neutral; the rail's caption is assembled at
# runtime purely from translated keys. The gesture hint stays English.
TITLE = "XelRay — CT 3D"
HINT = "drag to rotate · scroll to zoom · right-drag to pan · click an organ to show or hide it"
# The render's surround. Pure black, like the DICOM viewer's stage: any tint
# behind the meshes shifts how bright the tissue colours read.
STAGE = "#000000"
# Proper names stay untranslated; shown as a small footer credit line.
CREDIT = (
    "Segmentation: MONAI wholeBody_ct_segmentation (SegResNet) — Project MONAI, "
    "trained on the TotalSegmentator dataset (Wasserthal et al., Radiology AI 2023) · "
    "viewer: plotly.js · xelth.com/M"
)
OFFLINE_NOTE = "offline — language change needs a connection"

# label ids from the bundle's configs/metadata.json -> channel_def
VERTEBRAE = list(range(18, 42))
PELVIS_AND_RIBS = [88, 89, 90, 91, 92] + list(range(58, 88))

# i18n key -> ([label ids], hex color, opacity, shown by default)
GROUPS = [
    # Translucent so the tumour mesh sitting inside it stays visible.
    ("kidney_left", [3], "#D2521E", 0.50, True),
    ("kidney_right", [2], "#6E8CA0", 1.00, True),
    # The liver is by far the largest organ here; keep it translucent or it hides
    # everything retroperitoneal, the kidneys included.
    ("liver", [5], "#9B7B6B", 0.45, True),
    ("spleen", [1], "#7C8B6A", 0.70, True),
    ("pancreas", [10], "#C2A878", 0.90, True),
    ("aorta", [7], "#8A5560", 0.95, True),
    ("bones", VERTEBRAE + PELVIS_AND_RIBS, "#D0C9BA", 0.30, False),
]

TUMOUR_COLOR, TUMOUR_OPACITY = "#FF1F1F", 0.90


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def tr(key: str) -> str:
    """English label baked into the file; other languages are resolved in the browser."""
    return EN[key]


def index_to_physical(img: sitk.Image) -> np.ndarray:
    """3x3 matrix mapping (x, y, z) continuous voxel indices to physical mm."""
    d = np.asarray(img.GetDirection(), dtype=np.float64).reshape(3, 3)
    return d @ np.diag(img.GetSpacing())


def extract_mesh(mask_zyx: np.ndarray, img: sitk.Image, sigma: float, step: int = 1):
    """Marching-cubes surface of a binary mask, in physical LPS millimetres.

    The mask is padded (so surfaces close at the array border) and lightly
    gaussian-blurred, which is what turns voxel staircases into a smooth surface
    without a separate mesh-smoothing pass.
    """
    if not mask_zyx.any():
        return None
    pad = 2
    vol = np.pad(mask_zyx.astype(np.float32), pad)
    if sigma > 0:
        vol = ndi.gaussian_filter(vol, sigma)
    if vol.max() <= 0.5:  # blurred away entirely
        return None
    verts, faces, _, _ = measure.marching_cubes(vol, level=0.5, step_size=step)
    idx_zyx = verts - pad
    idx_xyz = idx_zyx[:, ::-1]  # array is (z, y, x); the affine wants (x, y, z)
    phys = np.asarray(img.GetOrigin()) + idx_xyz @ index_to_physical(img).T
    return phys, faces


def tumour_mask(out: Path, hu_max: float):
    """Estimate the tumour-bearing sub-volume *inside* the left-kidney label.

    The mass is not an unlabelled blob outside the kidney -- the model absorbed it
    into `kidney_left`. What separates it from parenchyma is enhancement: in the
    portal-venous phase normal cortex sits near 170 HU while this lesion is
    substantially lower. So: take the inferior part of the left-kidney label,
    keep voxels below `hu_max`, and return the largest connected component.
    """
    ct = sitk.ReadImage(str(out / "ct_orig.nii.gz"))
    lab = sitk.ReadImage(str(out / "pred_orig_space.nii.gz"))
    c = sitk.GetArrayFromImage(ct)
    a = sitk.GetArrayFromImage(lab)

    lk = a == 3
    if not lk.any():
        return None, None, {}
    zz = np.nonzero(lk)[0]
    zmid = (zz.min() + zz.max()) // 2

    cand = lk & (c < hu_max)
    cand[zmid + 8:] = False  # inferior segment only
    cand = ndi.binary_opening(cand, iterations=2)
    cc, n = ndi.label(cand)
    if n == 0:
        return None, None, {}
    biggest = np.bincount(cc.ravel())[1:].argmax() + 1
    m = ndi.binary_closing(cc == biggest, iterations=3)

    sp = ct.GetSpacing()
    z2, y2, x2 = np.nonzero(m)
    stats = {
        "hu_max": hu_max,
        "volume_ml": round(float(m.sum()) * float(np.prod(sp)) / 1000.0, 1),
        "extent_mm": [
            round(float((np.ptp(x2) + 1) * sp[0])),
            round(float((np.ptp(y2) + 1) * sp[1])),
            round(float((np.ptp(z2) + 1) * sp[2])),
        ],
        "mean_hu": int(c[m].mean()),
    }
    return m, ct, stats


# --- the page ----------------------------------------------------------------
#
# The figure is dropped into a page of our own rather than plotly's, because on
# a viewer the picture is the point: no plot title, no caption band, no legend,
# no footer annotations -- every one of those is a stripe of screen height taken
# away from the anatomy. All of the chrome lives instead in one narrow left rail
# that folds away, dressed like the XelRay DICOM viewer's rail
# (`crates/app_ui/styles.css`) so the two read as one product.
#
# CSS and JS are plain strings with a `__XR_DATA__` placeholder rather than
# f-strings: both are full of braces, and doubling every one of them is how
# these templates get quietly broken.

CSS = """
:root {
  /* xelth.com's felt, cream and gold, at the viewer's dimmer setting; the
     stage stays pure black so nothing tints the render. */
  --stage: #000000;
  --bg: #001611;
  --panel: #00241c;
  --panel-2: #033126;
  --line: #0d3b2f;
  --text: #cdbfa6;
  --bright: #ead8c1;
  --muted: #a09580;
  --faint: #998d75;
  --accent: #d9b36a;
  --warn: #e0836f;
  --rail: 228px;
  --radius: 7px;
  --mono: "Cascadia Mono", "Lucida Console", Consolas, ui-monospace, monospace;
  --sans: system-ui, -apple-system, "Segoe UI", Roboto, sans-serif;
}

* { box-sizing: border-box; }

html, body {
  margin: 0;
  height: 100%;
  /* The viewer owns the viewport; nothing here ever scrolls. */
  overflow: hidden;
  background: var(--bg);
  color: var(--text);
  font: 15px/1.55 var(--sans);
  font-weight: 450;
  -webkit-font-smoothing: antialiased;
}

.viewer { height: 100vh; display: flex; position: relative; }

.rail {
  width: var(--rail);
  flex: none;
  display: flex;
  flex-direction: column;
  border-right: 1px solid var(--line);
  background: var(--panel);
  font-size: 14px;
  overflow: hidden;
  user-select: none;
}
.viewer.folded .rail { display: none; }

.rail-top {
  display: flex;
  align-items: baseline;
  gap: 7px;
  padding: 8px 8px 8px 12px;
  border-bottom: 1px solid var(--line);
}
.logo {
  font-family: var(--mono);
  font-size: 15px;
  font-weight: 600;
  letter-spacing: -0.2px;
  color: var(--bright);
}
.logo::first-letter { color: var(--accent); }
.tag { flex: 1; font-family: var(--mono); font-size: 12.5px; color: var(--muted); }

.icon {
  align-self: center;
  width: 24px;
  height: 24px;
  display: grid;
  place-items: center;
  border: 1px solid transparent;
  border-radius: 5px;
  background: transparent;
  color: var(--muted);
  font-size: 15px;
  line-height: 1;
  cursor: pointer;
}
.icon:hover { background: var(--panel-2); color: var(--text); }

/* The strip that brings the rail back, in the corner the chevron just left. */
.rail-tab {
  position: absolute;
  top: 8px;
  left: 8px;
  z-index: 20;
  width: 30px;
  height: 30px;
  display: grid;
  place-items: center;
  border: 1px solid var(--line);
  border-radius: 6px;
  background: rgba(0, 36, 28, 0.82);
  color: var(--muted);
  font-size: 14px;
  cursor: pointer;
  opacity: 0.55;
  transition: opacity 0.12s;
}
.rail-tab:hover { opacity: 1; color: var(--text); }

.rail-body {
  flex: 1;
  overflow-y: auto;
  padding: 8px 8px 10px;
  scrollbar-width: thin;
  scrollbar-color: rgba(234, 216, 193, 0.15) transparent;
}

.caption {
  margin: 0 0 9px;
  padding: 0 3px;
  color: var(--warn);
  font-size: 12.5px;
  line-height: 1.45;
}

.organs { padding-top: 8px; border-top: 1px solid var(--line); }

/* One row per mesh: a dot in the mesh's own colour, the translated name, and
   the volume. The whole row is the visibility switch -- what used to be a
   plotly legend, which cost a floating panel over the anatomy. */
.organ {
  display: flex;
  align-items: baseline;
  gap: 8px;
  width: 100%;
  padding: 5px 8px;
  margin-bottom: 2px;
  border: 1px solid transparent;
  border-radius: var(--radius);
  background: transparent;
  color: var(--text);
  font: inherit;
  font-size: 13.5px;
  text-align: left;
  cursor: pointer;
}
.organ:hover { background: var(--panel-2); }
.organ .dot {
  flex: none;
  align-self: center;
  width: 9px;
  height: 9px;
  border-radius: 50%;
  background: var(--c);
}
.organ .nm { flex: 1; line-height: 1.35; overflow-wrap: anywhere; }
.organ .ml { flex: none; font-family: var(--mono); font-size: 12px; color: var(--muted); }
/* Hidden reads as drained, not as absent: the dot keeps its hue as a ring so
   the row is still identifiable, and one click brings the mesh back. */
.organ.off { color: var(--faint); }
.organ.off .dot { background: transparent; box-shadow: inset 0 0 0 1px var(--c); opacity: 0.65; }
.organ.off .ml { color: var(--faint); }

.rail-foot { padding: 8px 11px 10px; border-top: 1px solid var(--line); }

.rail-links { display: flex; align-items: center; justify-content: center; }
/* Dressed as one of the links, not as a control competing with the render. */
.lang {
  appearance: none;
  padding: 1px 3px;
  border: none;
  background: transparent;
  color: var(--muted);
  font-family: var(--mono);
  font-size: 12.5px;
  cursor: pointer;
}
.lang:hover { color: var(--accent); }
.lang:focus-visible { outline: 1px solid var(--accent); border-radius: 3px; }
/* The popup is drawn by the OS, so it needs real colours. */
.lang option { background: var(--panel); color: var(--text); }

.hint { margin: 8px 0 0; color: var(--muted); font-size: 11.5px; line-height: 1.45; }
.credit { margin: 8px 0 0; color: var(--faint); font-size: 9.5px; line-height: 1.4; }
.note {
  margin: 8px 0 0;
  color: var(--warn);
  font-size: 11.5px;
  line-height: 1.4;
  opacity: 0;
  transition: opacity 0.25s;
}
.note.on { opacity: 1; }

/* Everything right of the rail is the render, edge to edge. */
.stage {
  position: relative;
  flex: 1;
  min-width: 0;
  min-height: 0;
  overflow: hidden;
  background: var(--stage);
}
/* plotly wraps its div (and its own <script> tags) in one anonymous div. */
.stage > div { width: 100%; height: 100%; }

@media (max-width: 640px) { :root { --rail: 180px; } }
"""

JS = r"""
(function () {
  var D = __XR_DATA__;
  var viewer = document.getElementById("xr-viewer");
  var sel = document.getElementById("xr-lang");
  var note = document.getElementById("xr-note");
  var caption = document.getElementById("xr-caption");
  var tab = document.getElementById("xr-show");
  var current = "en";
  var noteTimer = null;
  var busy = false;
  var rows = [];

  function plot() { return document.querySelector(".plotly-graph-div"); }

  function ls(op, k, v) {
    try {
      if (op === "get") return localStorage.getItem(k);
      localStorage.setItem(k, v);
    } catch (e) {}
    return null;
  }

  function say(msg) {
    note.textContent = msg;
    note.classList.add("on");
    clearTimeout(noteTimer);
    noteTimer = setTimeout(function () { note.classList.remove("on"); }, 4000);
  }

  /* ---- rail ------------------------------------------------------------- */

  function buildRail() {
    var host = document.getElementById("xr-organs");
    D.keys.forEach(function (k, n) {
      var b = document.createElement("button");
      b.type = "button";
      b.className = "organ";
      b.style.setProperty("--c", D.colors[n]);
      b.innerHTML =
        '<span class="dot"></span><span class="nm"></span><span class="ml"></span>';
      b.addEventListener("click", function () { toggle(n); });
      host.appendChild(b);
      rows.push(b);
    });
  }

  function shown(gd, n) {
    var v = gd.data[n].visible;
    return v !== false && v !== "legendonly";
  }

  // The rows follow the figure rather than their own state, so a toggle that
  // plotly refuses (or a re-plot) can never leave the rail lying.
  function paint() {
    var gd = plot();
    if (!gd || !gd.data) return;
    rows.forEach(function (b, n) {
      var on = shown(gd, n);
      b.classList.toggle("off", !on);
      b.setAttribute("aria-pressed", on ? "true" : "false");
    });
  }

  function toggle(n) {
    var gd = plot();
    if (!gd || !window.Plotly || busy) return;
    Plotly.restyle(gd, { visible: !shown(gd, n) }, [n]).then(paint, paint);
  }

  function retext(S) {
    D.keys.forEach(function (k, n) {
      rows[n].querySelector(".nm").textContent = S[k];
      rows[n].querySelector(".ml").textContent =
        (k === "tumor_region" ? "~" : "") + D.vols[n] + " " + S.unit_ml;
    });
    caption.textContent =
      S.kidney_left + ": " + S.tumor_region + " · " + S.ai_disclaimer;
  }

  function fold(on) {
    viewer.classList.toggle("folded", on);
    tab.hidden = !on;
    ls("set", "xr-rail", on ? "0" : "1");
    var gd = plot();
    if (gd && window.Plotly) Plotly.Plots.resize(gd);
  }

  /* ---- language --------------------------------------------------------- */

  // Plotly locale dictionaries are keyed by plotly's English strings, so D.ui
  // maps our key -> that English string -> the translation (English if absent).
  function makeConfig(lang, U) {
    var c = {};
    Object.keys(D.config).forEach(function (k) { c[k] = D.config[k]; });
    c.locale = lang;
    if (lang !== "en") {
      var dict = {};
      Object.keys(D.ui).forEach(function (k) { dict[D.ui[k]] = U[k]; });
      c.locales = {};
      c.locales[lang] = { dictionary: dict };
    }
    return c;
  }

  // Belt and braces: after the re-plot, make each gl3d trace wrapper (whose
  // `.name` is what the hover box prints) agree with its trace again.
  function syncHoverNames(gd) {
    try {
      var scene = gd._fullLayout && gd._fullLayout.scene && gd._fullLayout.scene._scene;
      if (!scene || !scene.traces) return;
      Object.keys(scene.traces).forEach(function (uid) {
        for (var i = 0; i < gd._fullData.length; i++) {
          if (gd._fullData[i].uid === uid) {
            scene.traces[uid].name = gd._fullData[i].name;
            return;
          }
        }
      });
    } catch (e) {}
  }

  // `bundle` is {organ: {...}, ui: {...}}; any key the server omits for a
  // given language falls back to English rather than rendering blank.
  function render(lang, bundle) {
    var gd = plot();
    if (!gd || !window.Plotly || !gd.data) return;
    var organ = bundle.organ || {};
    var ui = bundle.ui || {};
    var S = {}, U = {};
    Object.keys(D.en).forEach(function (k) { S[k] = organ[k] || D.en[k]; });
    Object.keys(D.ui).forEach(function (k) { U[k] = ui[k] || D.ui[k]; });

    // The rail is plain DOM and retexts instantly.
    retext(S);
    document.documentElement.lang = lang;
    current = lang;
    sel.value = lang;
    ls("set", "xr-lang", lang);

    // The figure needs a re-plot: the hover box prints the gl3d wrapper's
    // name and the modebar tooltips read the config's locale, and a restyle
    // refreshes neither.
    D.keys.forEach(function (k, n) {
      var s = S[k];
      if (D.vols[n] != null) {
        s += k === "tumor_region"
           ? " ~" + D.vols[n] + " " + S.unit_ml
           : " (" + D.vols[n] + " " + S.unit_ml + ")";
      }
      if (gd.data[n]) gd.data[n].name = s;
    });

    // Keep the view the reader is on: the re-plot would otherwise snap back to
    // the camera baked into the figure. The scene's own camera is the only
    // reliable source -- `layout.scene.camera` is only written back on the
    // relayout that ends a drag, so it can lag a rotation by a whole gesture.
    var cam = null;
    try {
      var scene = gd._fullLayout.scene;
      cam = (scene._scene && scene._scene.getCamera()) || scene.camera;
    } catch (e) {}
    if (cam) gd.layout.scene.camera = cam;

    busy = true;
    Plotly.react(gd, gd.data, gd.layout, makeConfig(lang, U)).then(function () {
      syncHoverNames(gd);
      paint();
      busy = false;
    }, function () { busy = false; });
  }

  function cached(lang) {
    var raw = ls("get", "xr-i18n-" + lang);
    if (!raw) return null;
    try {
      var v = JSON.parse(raw);
      if (!v || typeof v !== "object") return null;
      // Tolerate the older cache shape, which was a flat map of organ keys.
      return v.organ ? v : { organ: v, ui: {} };
    } catch (e) { return null; }
  }

  function select(lang, quiet) {
    if (D.langs.indexOf(lang) < 0) lang = "en";
    if (busy) { sel.value = current; return; }
    if (lang === "en") { render("en", { organ: D.en, ui: D.ui }); return; }

    var hit = cached(lang);
    if (hit) { render(lang, hit); return; }

    fetch(D.url + lang, { mode: "cors", credentials: "omit" })
      .then(function (r) {
        if (!r.ok) throw new Error("HTTP " + r.status);
        return r.json();
      })
      .then(function (all) {
        var bundle = { organ: {}, ui: {} };
        Object.keys(all).forEach(function (k) {
          if (k.indexOf(D.prefix) === 0) {
            bundle.organ[k.slice(D.prefix.length)] = all[k];
          } else if (k.indexOf(D.prefixUi) === 0) {
            bundle.ui[k.slice(D.prefixUi.length)] = all[k];
          }
        });
        // The viewer3d keys may not be deployed yet; the organ names are what
        // makes a language worth switching to, so only those are required.
        if (!Object.keys(bundle.organ).length) throw new Error("no " + D.prefix + " keys");
        ls("set", "xr-i18n-" + lang, JSON.stringify(bundle));
        render(lang, bundle);
      })
      .catch(function () {
        sel.value = current;          // keep whatever is on screen
        if (!quiet) say(D.offline);
      });
  }

  /* ---- boot ------------------------------------------------------------- */

  var saved = ls("get", "xr-lang");
  var guess = (navigator.language || "en").slice(0, 2).toLowerCase();
  var initial = D.langs.indexOf(saved) >= 0 ? saved
              : (D.langs.indexOf(guess) >= 0 ? guess : "en");

  sel.addEventListener("change", function () { select(sel.value, false); });
  document.getElementById("xr-hide").addEventListener("click", function () { fold(true); });
  tab.addEventListener("click", function () { fold(false); });

  // `s` and `Tab` fold the rail, the same two keys the DICOM viewer uses.
  document.addEventListener("keydown", function (e) {
    if (e.ctrlKey || e.metaKey || e.altKey) return;
    var t = e.target;
    if (t && (t.tagName === "SELECT" || t.tagName === "INPUT" || t.isContentEditable)) return;
    if (e.key === "s" || e.key === "S" || e.key === "Tab") {
      e.preventDefault();
      fold(!viewer.classList.contains("folded"));
    }
  });

  function boot() {
    buildRail();
    retext(D.en);                     // English is baked into the figure
    paint();
    sel.value = "en";
    if (ls("get", "xr-rail") === "0") fold(true);
    if (initial !== "en") select(initial, true);
  }
  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", boot);
  } else {
    boot();
  }
})();
"""


def page(
    plot_div: str,
    trace_keys: list[str],
    volumes: list[float | None],
    colors: list[str],
) -> str:
    """The whole document: left rail, then the figure filling everything else.

    English is embedded and always works with no network. Any other language is
    fetched once from the XelRay i18n endpoint (which sends
    `Access-Control-Allow-Origin: *`, so this works even from a `file://` page,
    where the Origin is `null`) and then cached in localStorage together with the
    choice -- a later offline reopen restores that language from cache. If a fetch
    for an uncached language fails, the current language is kept and a small note
    appears in the rail; nothing else breaks.

    One bundle carries two key spaces: `xelray.organ.*` for the meshes and
    `xelray.viewer3d.*` for plotly's own modebar tooltips. Either may be missing
    per language; every key falls back to the embedded English individually.

    The rail retexts as plain DOM, but the figure is re-applied with
    `Plotly.react` and a fresh config, for two reasons:

    * modebar tooltips are translated through plotly's `locale`/`locales` config,
      which is only read when the plot's context is rebuilt; and
    * `restyle({name})` does *not* reach the hover label. A gl3d scene keeps its
      own wrapper per trace and the hover box reads that wrapper's `.name`,
      which only a scene re-plot refreshes -- a name edit is an
      `editType: "style"` change, and `doTraceStyle` redraws the (now absent)
      legend without touching the WebGL scene. The camera is carried across the
      re-plot explicitly so the reader's view survives the switch.
    """
    options = "".join(f'<option value="{lg}">{LANG_NAMES[lg]}</option>' for lg in LANGS)
    payload = json.dumps(
        {
            "langs": LANGS,
            "en": EN,
            "ui": MODEBAR,
            "keys": trace_keys,
            "colors": colors,
            # Pre-rounded the way the baked English labels are, so a language
            # switch never nudges a volume by 1 ml (python and JS disagree on
            # halves: format(96.5, ".0f") is 96, Math.round(96.5) is 97).
            "vols": [None if v is None else int(f"{v:.0f}") for v in volumes],
            "url": I18N_URL,
            "prefix": I18N_PREFIX,
            "prefixUi": I18N_PREFIX_UI,
            "config": PLOT_CONFIG,
            "offline": OFFLINE_NOTE,
        },
        ensure_ascii=True,  # \uXXXX escapes: safe whatever encoding the file is served as
    )
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{TITLE}</title>
<style>{CSS}</style>
</head>
<body>
<div class="viewer" id="xr-viewer">
  <aside class="rail">
    <div class="rail-top">
      <span class="logo">XelRay</span><span class="tag">CT 3D</span>
      <button class="icon" id="xr-hide" title="Hide panel (s)" aria-label="Hide panel">&lsaquo;</button>
    </div>
    <div class="rail-body">
      <p class="caption" id="xr-caption"></p>
      <div class="organs" id="xr-organs"></div>
    </div>
    <div class="rail-foot">
      <div class="rail-links">
        <select class="lang" id="xr-lang" aria-label="Language">{options}</select>
      </div>
      <p class="hint">{HINT}</p>
      <p class="credit">{CREDIT}</p>
      <p class="note" id="xr-note" role="status"></p>
    </div>
  </aside>
  <button class="rail-tab" id="xr-show" hidden title="Show panel (s)" aria-label="Show panel">&#9776;</button>
  <div class="stage">{plot_div}</div>
</div>
<script>{JS.replace("__XR_DATA__", payload)}</script>
</body>
</html>
"""


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="out")
    ap.add_argument("--space", default="model", choices=["model", "orig"])
    ap.add_argument("--tumour-hu", type=float, default=150.0)
    ap.add_argument("--no-tumour", action="store_true")
    args = ap.parse_args()

    import plotly.graph_objects as go

    out = Path(args.out)
    src = "pred_model_space.nii.gz" if args.space == "model" else "pred_orig_space.nii.gz"
    img = sitk.ReadImage(str(out / src))
    lab = sitk.GetArrayFromImage(img)
    vox_ml = float(np.prod(img.GetSpacing())) / 1000.0
    # 3 mm voxels need a wider blur (in voxels) than 0.76 mm ones do.
    sigma = 0.9 if args.space == "model" else 2.0
    step = 1 if args.space == "model" else 2
    log(f"meshing from {src} {lab.shape} at {img.GetSpacing()} mm")

    traces: list = []
    trace_keys: list[str] = []
    trace_vols: list[float | None] = []
    trace_colors: list[str] = []
    stats: dict = {}

    for key, ids, color, opacity, visible in GROUPS:
        mask = np.isin(lab, ids)
        if not mask.any():
            log(f"  skip {key}: no voxels")
            continue
        got = extract_mesh(mask, img, sigma, step)
        if got is None:
            continue
        v, f = got
        ml = round(float(mask.sum()) * vox_ml, 1)
        stats[key] = {"en": tr(key), "volume_ml": ml,
                      "vertices": len(v), "faces": len(f)}
        log(f"  {key:13s} {ml:7.1f} ml  {len(v):6d} verts  {len(f):6d} faces")
        trace_keys.append(key)
        trace_vols.append(ml)
        trace_colors.append(color)
        traces.append(
            go.Mesh3d(
                x=v[:, 0], y=-v[:, 1], z=v[:, 2],
                i=f[:, 0], j=f[:, 1], k=f[:, 2],
                color=color, opacity=opacity,
                name=f"{tr(key)} ({ml:.0f} ml)",
                # There is no legend any more -- the rail's organ rows are the
                # visibility switches, so hidden means hidden, not "legendonly".
                visible=visible,
                flatshading=False, hoverinfo="name",
                lighting=dict(ambient=0.55, diffuse=0.85, specular=0.18, roughness=0.55),
                lightposition=dict(x=200, y=-400, z=600),
            )
        )

    if not args.no_tumour:
        m, ct, tstats = tumour_mask(out, args.tumour_hu)
        if m is not None:
            got = extract_mesh(m, ct, sigma=2.0, step=2)
            if got is not None:
                v, f = got
                tstats.update(vertices=len(v), faces=len(f), en=tr("tumor_region"))
                stats["tumor_region"] = tstats
                log(
                    f"  {'tumor_region':13s} {tstats['volume_ml']:7.1f} ml  "
                    f"{tstats['extent_mm']} mm  meanHU {tstats['mean_hu']}  "
                    f"{len(v)} verts {len(f)} faces"
                )
                trace_keys.append("tumor_region")
                trace_vols.append(tstats["volume_ml"])
                trace_colors.append(TUMOUR_COLOR)
                traces.append(
                    go.Mesh3d(
                        x=v[:, 0], y=-v[:, 1], z=v[:, 2],
                        i=f[:, 0], j=f[:, 1], k=f[:, 2],
                        color=TUMOUR_COLOR, opacity=TUMOUR_OPACITY,
                        name=f"{tr('tumor_region')} ~{tstats['volume_ml']:.0f} ml",
                        visible=True, flatshading=False,
                        hoverinfo="name",
                        lighting=dict(ambient=0.75, diffuse=0.6, specular=0.1),
                    )
                )

    fig = go.Figure(data=traces)
    axis = dict(
        showgrid=False, zeroline=False, showticklabels=False, visible=False,
        showbackground=False,
    )
    # No title, no caption, no legend, no annotations and no margins: the title
    # and caption are the rail's job, the legend is the rail's organ list, and
    # every band of chrome plotly draws is a band of anatomy the reader loses.
    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor=STAGE,
        showlegend=False,
        scene=dict(
            xaxis=axis, yaxis=axis, zaxis=axis,
            aspectmode="data",
            bgcolor=STAGE,
            # Plot axes are (patient-left, anterior, superior). The kidneys are
            # retroperitoneal, so the informative default is a posterior-oblique
            # view from the patient's left -- straight onto the affected kidney.
            camera=dict(eye=dict(x=1.35, y=-1.55, z=0.5), up=dict(x=0, y=0, z=1)),
        ),
        margin=dict(l=0, r=0, t=0, b=0),
    )

    plot_div = fig.to_html(
        include_plotlyjs=True, full_html=False,
        config=dict(PLOT_CONFIG, locale="en"),
    )
    html = page(plot_div, trace_keys, trace_vols, trace_colors)

    path = out / "render3d.html"
    path.write_text(html, encoding="utf-8")
    (out / "mesh_stats.json").write_text(
        json.dumps(stats, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    log(
        f"wrote {path} ({path.stat().st_size / 1e6:.1f} MB), "
        f"{len(traces)} meshes, {len(LANGS)} languages"
    )


if __name__ == "__main__":
    main()
