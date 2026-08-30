# XelRay

**Free, private, in-browser DICOM viewer — your medical images never leave your computer.**

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-WebAssembly-orange.svg)](https://www.rust-lang.org/)
[![Leptos](https://img.shields.io/badge/Leptos-0.6-red.svg)](https://leptos.dev/)

**Live: <https://xelth.com/M/xelray/>**

You get a CD or a USB stick from the hospital, and on it a `DICOM` folder full of
files with no extension. The bundled viewer is a Windows-only executable from
2009, and every "free online DICOM viewer" wants you to upload your scan to
somebody's server first.

XelRay is the other option. Drop the folder into the page. The DICOM parser is
Rust compiled to WebAssembly and runs inside your browser tab — there is no
backend, no upload, no account. Close the tab and nothing is left anywhere.

## Screenshot

<!-- TODO: replace with a real screenshot of a loaded CT series -->
![XelRay viewing a CT series](docs/screenshot.png)

## Features

- **Drag & drop a whole folder** — straight from the CD, or pick it with the
  file dialog (`webkitdirectory`). Sub-folders are walked automatically.
- **Series detection** — instances are grouped by `SeriesInstanceUID` and
  listed in a sidebar with modality and image count.
- **Correct slice order** — slices are sorted by their projection onto the
  slice normal derived from `ImageOrientationPatient`, so oblique acquisitions
  come out right; `InstanceNumber` is the fallback.
- **Window / level** — presets for soft tissue (400/40), lung (1500/−600),
  bone (1800/400) and brain (80/40), plus the usual left-drag adjustment
  (horizontal = width, vertical = level). Defaults come from the file's own
  `WindowWidth`/`WindowCenter`.
- **Hounsfield-correct pixels** — `RescaleSlope`/`RescaleIntercept` are applied,
  so the CT presets mean what they say.
- **All image, no chrome** — every control lives in one narrow left rail that
  folds away with `S`, giving the scan the entire viewport. The text over the
  image hides too, with `O`.
- **Keyboard first** — everything is reachable without the mouse, using only
  keys a laptop actually has (see below). `?` brings up the cheat sheet.
- **Trackpad native** — two-finger scroll steps through the stack with the
  deltas accumulated, so one swipe moves a few images rather than thirty;
  pinch zooms smoothly. Touch screens get swipe and pinch too.
- **Bounded memory** — a 500 MB hospital CD opens in a 32-bit WebAssembly
  heap, because the study is never held in memory. Loading reads a 64 KB
  prefix of each file, keeps a few hundred bytes of metadata and drops the
  rest; pixels are decoded one image at a time into a byte-budgeted LRU that
  also warms the neighbours so scrolling stays smooth.
- **Graceful about what it cannot decode** — an unsupported transfer syntax
  produces a per-series warning, not a crash.

## Keyboard

Bindings assume a laptop: no numpad, and nothing that needs `Fn`. `Home`,
`End`, `PageUp` and `PageDown` work as well, but never as the only way to do
something.

| Keys | Action |
| --- | --- |
| `↑` `↓` | Previous / next image |
| `Shift+↑` `Shift+↓` | Jump 10 images |
| `g` `G` | First / last image |
| `←` `→` or `[` `]` | Previous / next series |
| `1` `2` `3` `4` | Soft tissue · Lung · Bone · Brain |
| `=` `-` | Zoom in / out |
| `0` or `f` | Fit to window, undo zoom |
| `s` or `Tab` | Show / hide the panel |
| `o` | Show / hide the text over the image |
| `?` or `h` | Shortcut list |
| `Esc` | Close the shortcut list |

Mouse: wheel steps images, ctrl+wheel zooms, left-drag sets window/level,
middle-drag pans, double-click fits. Files can be dropped anywhere on the
window.

## Build

Requires a Rust toolchain, the `wasm32-unknown-unknown` target and
[Trunk](https://trunkrs.dev/):

```sh
rustup target add wasm32-unknown-unknown
cargo install trunk
```

Then, from `crates/app_ui`:

```sh
trunk serve                       # dev server on http://localhost:8080
trunk build --release             # production bundle in crates/app_ui/dist/
trunk build --release --public-url /   # if serving from a domain root
```

The default `public_url` is `/M/xelray/`, matching where the app is deployed on
xelth.com.

> On Windows with Git Bash, prefix Trunk invocations with `MSYS_NO_PATHCONV=1`
> so MSYS does not rewrite `--public-url` into a Windows path.

Run the parser tests on the native target:

```sh
cargo test -p xelray_core
```

## Architecture

```
crates/
  xelray_core/   pure Rust, no browser APIs — parsing, series grouping,
                 slice sorting, pixel decoding. Unit-tested natively.
  app_ui/        Leptos 0.6 CSR front end, built by Trunk to WebAssembly.
```

The split is deliberate: `xelray_core` takes `(filename, bytes)` pairs and
returns a sorted `Study`, so the whole DICOM path can be tested with `cargo
test` on a normal desktop target, and the exact same code then runs in the
browser. It builds on the excellent [dicom-rs](https://github.com/Enet4/dicom-rs)
crates.

Rendering is a plain 2D canvas: the window transform maps modality values to
8-bit grey into a reused `ImageData` buffer, and zoom/pan are a CSS transform
on the canvas element, so dragging never triggers a re-decode.

### Memory

The interesting constraint is that a wasm32 heap cannot hold a real study. A
1000-image CT is ~500 MB encoded and ~1 GB decoded as `f32`. So nothing is
ever loaded whole:

- The browser's `File` objects stay on the JS side. They are handles to bytes
  on disk and cost nothing until read.
- Indexing reads a 64 KB prefix per file and parses only up to `PixelData`
  (`OpenFileOptions::read_until`), so the pixels are never even allocated.
  What is retained is a ~130-byte `Instance` per image. A file whose header
  runs past the prefix — rare — falls back to one transient full read.
- Displaying an image reads that one file, decodes it, and puts the result in
  a `SliceCache`: an LRU budgeted in *bytes* (48 MB by default), because slice
  size varies by an order of magnitude across modalities. Neighbours ±3 are
  prefetched; navigating bumps a generation counter so work the user has
  scrolled past is abandoned instead of completed.

For a 1000-image 512² CT that comes to roughly **60–70 MB**: ~130 KB of
index, 48 MB of decoded slices, and a transient megabyte or two for the file
being decoded. The ceiling is set by the cache, not by the study.

### Format support

Uncompressed (implicit and explicit VR), RLE, deflated and JPEG-compressed
transfer syntaxes decode. JPEG 2000 and JPEG XL do not: their decoders in
dicom-rs are C libraries that do not build for `wasm32-unknown-unknown`. Those
series are flagged in the UI rather than failing silently.

Multi-frame instances currently show their first frame only; the stack is
built from single-frame instances, which is what CT and MR studies on a
hospital CD look like.

## Privacy

There is no server component in this repository, and the app makes no network
requests after the page itself has loaded. Files are read with the browser's
File API into WebAssembly memory and discarded when the tab closes.

XelRay is a viewer, not a diagnostic device. It is not certified medical
software — do not use it to make clinical decisions.

## Part of the xelth.com medical tools family

XelRay is the first tool in the `/M/` (medical) section of
[xelth.com](https://xelth.com). More to follow.

## License

MIT — see [LICENSE](LICENSE).

---

## По-русски

**XelRay — бесплатный DICOM-просмотрщик, работающий прямо в браузере. Ваши
снимки никуда не отправляются.**

Вам выдали в больнице диск с папкой `DICOM`, а просмотрщик на диске — старая
программа только для Windows. Онлайн-сервисы требуют сначала загрузить
исследование на чужой сервер.

XelRay работает иначе: перетащите папку в окно браузера. Разбор DICOM написан
на Rust и скомпилирован в WebAssembly — он выполняется внутри вкладки. Сервера
нет вообще, ничего никуда не загружается, регистрация не нужна. Закрыли
вкладку — не осталось ничего.

Что умеет: распознаёт серии, правильно сортирует срезы, применяет
`RescaleSlope`/`RescaleIntercept` (то есть шкалу Хаунсфилда), даёт пресеты окна
(мягкие ткани, лёгкие, кости, мозг) и настройку окна мышью, прокрутку колесом
и двумя пальцами по тачпаду, масштабирование и перемещение. Большие
исследования на 500–1000 срезов загружаются постепенно, со счётчиком.

Всё управление доступно с клавиатуры: `↑`/`↓` — срезы, `Shift+↑`/`↓` — по
десять, `g`/`G` — первый и последний, `←`/`→` — серии, `1`–`4` — пресеты окна,
`=`/`-` — масштаб, `0` — вписать в окно, `s` — убрать панель (снимок займёт
весь экран), `o` — убрать подписи, `?` — список горячих клавиш.

Открыть: <https://xelth.com/M/xelray/>

XelRay — просмотрщик, а не медицинское изделие. Не используйте его для
постановки диагноза.
