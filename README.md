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
- **Navigation** — mouse wheel or arrow keys to scroll the stack, a scrub bar,
  ctrl+wheel or +/− to zoom, middle-drag (or the *Pan* tool) to pan.
- **Incremental loading** — a 1000-image study streams in with a live counter
  instead of freezing the tab; pixel data is decoded only for the slice you
  are actually looking at.
- **Graceful about what it cannot decode** — an unsupported transfer syntax
  produces a per-series warning, not a crash.

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
8-bit grey into an `ImageData`, and zoom/pan are a CSS transform on the canvas
element, so dragging never triggers a re-decode.

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
(мягкие ткани, лёгкие, кости, мозг) и настройку окна мышью, прокрутку колесом,
масштабирование и перемещение. Большие исследования на 500–1000 срезов
загружаются постепенно, со счётчиком.

Открыть: <https://xelth.com/M/xelray/>

XelRay — просмотрщик, а не медицинское изделие. Не используйте его для
постановки диагноза.
