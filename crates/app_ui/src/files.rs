//! Getting bytes out of the browser — as little of them as possible.
//!
//! Two entry points matter: a `<input type=file webkitdirectory>` pick, which
//! already hands over a flat `FileList`, and a drag-and-drop of a folder,
//! which hands over directory *entries* that have to be walked.
//!
//! The `File` objects themselves are kept, not their contents. A `File` is a
//! handle to bytes still sitting on disk, costing nothing until read, which
//! is what lets the viewer hold a 500 MB study open in a 32-bit heap. Reads
//! are deliberately narrow: [`read_prefix`] for indexing, [`read_all`] only
//! for the one image being displayed.

use js_sys::{Array, Uint8Array};
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{
    DataTransfer, File, FileList, FileSystemDirectoryEntry, FileSystemDirectoryReader,
    FileSystemEntry, FileSystemFileEntry,
};

/// Flatten a `FileList` (from the file/folder picker) into a plain vector.
pub fn from_file_list(list: &FileList) -> Vec<File> {
    (0..list.length()).filter_map(|i| list.get(i)).collect()
}

// ---------------------------------------------------------------------------
// Folder picking
//
// `<input webkitdirectory>` makes Chrome ask "Upload 995 files to this site?"
// — in the user's own language, with the word *upload* in it. XelRay uploads
// nothing, and being told otherwise by the browser, every single time, is
// worse than a papercut: it contradicts the one promise the whole tool makes.
//
// The File System Access API asks instead to *view* files, once, which is
// both milder and true. It is Chromium-only, so the input stays as the
// fallback for Firefox and Safari.
//
// This is a JS shim rather than `web_sys`'s bindings because those are behind
// `--cfg web_sys_unstable_apis`, a flag that would have to be set for every
// build of the whole crate graph. Twenty lines of JS keeps the flag out of
// the project entirely, and directory traversal is more natural in a language
// with `for await` than through hand-rolled async-iterator plumbing.
// ---------------------------------------------------------------------------

#[wasm_bindgen(inline_js = r#"
export function has_directory_picker() {
    return typeof window.showDirectoryPicker === 'function';
}

// Resolves to an Array of File, or null if the user cancelled or the picker
// could not be opened. Never rejects: the caller treats every failure the
// same way, by doing nothing.
export async function pick_directory() {
    if (typeof window.showDirectoryPicker !== 'function') return null;

    let root;
    try {
        // Called before the first await, so the click's transient activation
        // is still live.
        root = await window.showDirectoryPicker({ mode: 'read' });
    } catch (e) {
        // AbortError is the user closing the dialog — the expected outcome
        // half the time, and not a failure.
        return null;
    }

    // Hospital CDs nest: PA000000/ST000000/SE000000/IM000000. Iterative so a
    // deep tree cannot blow the stack.
    const files = [];
    const queue = [root];
    while (queue.length > 0) {
        const dir = queue.pop();
        try {
            for await (const entry of dir.values()) {
                if (entry.kind === 'file') {
                    // One unreadable entry must not lose the other 994.
                    try { files.push(await entry.getFile()); } catch (e) {}
                } else if (entry.kind === 'directory') {
                    queue.push(entry);
                }
            }
        } catch (e) {}
    }
    return files;
}
"#)]
extern "C" {
    #[wasm_bindgen(js_name = has_directory_picker)]
    fn has_directory_picker_js() -> bool;

    #[wasm_bindgen(js_name = pick_directory)]
    fn pick_directory_js() -> js_sys::Promise;
}

/// Whether this browser can open a folder without the upload warning.
///
/// Feature detection, deliberately not a user-agent test: the API arrives in
/// browsers on their own schedule and can be disabled by policy.
pub fn has_directory_picker() -> bool {
    has_directory_picker_js()
}

/// Open the native folder picker.
///
/// Call this *synchronously* from the click handler — the returned promise
/// can be awaited later, but the picker must be opened while the user's
/// activation is still live or the browser will refuse it.
pub fn open_directory_picker() -> js_sys::Promise {
    pick_directory_js()
}

/// Resolve the picker's promise into files.
///
/// `None` means cancelled or unavailable, and is not an error worth showing
/// anyone.
pub async fn awaited_picker_files(promise: js_sys::Promise) -> Option<Vec<File>> {
    let value = JsFuture::from(promise).await.ok()?;
    let array = value.dyn_into::<Array>().ok()?;
    Some(
        array
            .iter()
            .filter_map(|v| v.dyn_into::<File>().ok())
            .collect(),
    )
}

/// Collect every file in a drop, descending into dropped folders.
///
/// Chrome and Firefox expose dropped directories only through
/// `webkitGetAsEntry`; `DataTransfer::files` would give us the folder itself,
/// which reads as an empty blob. We therefore try the entry API first and
/// fall back to the flat file list for browsers that do not implement it.
pub async fn from_data_transfer(dt: &DataTransfer) -> Vec<File> {
    let items = dt.items();
    let mut roots: Vec<FileSystemEntry> = Vec::new();
    for i in 0..items.length() {
        if let Some(entry) = items.get(i).and_then(|it| it.webkit_get_as_entry().ok().flatten()) {
            roots.push(entry);
        }
    }

    if roots.is_empty() {
        return dt.files().map(|l| from_file_list(&l)).unwrap_or_default();
    }

    // Breadth-first rather than recursive: an async fn cannot call itself
    // without boxing, and a work list is both cheaper and easier to read.
    let mut out = Vec::new();
    let mut queue = roots;
    while let Some(entry) = queue.pop() {
        if entry.is_file() {
            if let Ok(fe) = entry.dyn_into::<FileSystemFileEntry>() {
                if let Some(file) = entry_file(&fe).await {
                    out.push(file);
                }
            }
        } else if entry.is_directory() {
            if let Ok(de) = entry.dyn_into::<FileSystemDirectoryEntry>() {
                queue.extend(read_dir(&de).await);
            }
        }
    }
    out
}

/// Resolve a file entry to its `File` (callback API wrapped as a promise).
async fn entry_file(entry: &FileSystemFileEntry) -> Option<File> {
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let ok = Closure::once_into_js(move |f: JsValue| {
            let _ = resolve.call1(&JsValue::NULL, &f);
        });
        let err = Closure::once_into_js(move |e: JsValue| {
            let _ = reject.call1(&JsValue::NULL, &e);
        });
        entry.file_with_callback_and_callback(ok.unchecked_ref(), err.unchecked_ref());
    });
    JsFuture::from(promise).await.ok()?.dyn_into::<File>().ok()
}

/// Read a directory completely.
///
/// `readEntries` is allowed to return a partial batch, so it must be called
/// until it yields an empty array — a folder of 500 CT slices comes back in
/// chunks of 100 in Chrome.
async fn read_dir(dir: &FileSystemDirectoryEntry) -> Vec<FileSystemEntry> {
    let reader = dir.create_reader();
    let mut all = Vec::new();
    loop {
        let batch = read_entries_once(&reader).await;
        if batch.is_empty() {
            return all;
        }
        all.extend(batch);
    }
}

async fn read_entries_once(reader: &FileSystemDirectoryReader) -> Vec<FileSystemEntry> {
    let promise = js_sys::Promise::new(&mut |resolve, reject| {
        let ok = Closure::once_into_js(move |v: JsValue| {
            let _ = resolve.call1(&JsValue::NULL, &v);
        });
        let err = Closure::once_into_js(move |e: JsValue| {
            let _ = reject.call1(&JsValue::NULL, &e);
        });
        let _ = reader.read_entries_with_callback_and_callback(ok.unchecked_ref(), err.unchecked_ref());
    });

    let Ok(value) = JsFuture::from(promise).await else {
        return Vec::new();
    };
    let Ok(array) = value.dyn_into::<Array>() else {
        return Vec::new();
    };
    array
        .iter()
        .filter_map(|v| v.dyn_into::<FileSystemEntry>().ok())
        .collect()
}

/// Read the first `len` bytes of a file.
///
/// `Blob::slice` does not copy anything: it returns a view that the browser
/// reads from disk when awaited, so indexing a study touches a few kilobytes
/// per image rather than half a megabyte.
pub async fn read_prefix(file: &File, len: usize) -> Option<Vec<u8>> {
    let end = len.min(file.size() as usize) as i32;
    let blob = file.slice_with_i32_and_i32(0, end).ok()?;
    let buffer = JsFuture::from(blob.array_buffer()).await.ok()?;
    Some(Uint8Array::new(&buffer).to_vec())
}

/// Read a whole file. Used for exactly one image at a time.
pub async fn read_all(file: &File) -> Option<Vec<u8>> {
    let buffer = JsFuture::from(file.array_buffer()).await.ok()?;
    Some(Uint8Array::new(&buffer).to_vec())
}

/// `DICOMDIR` indexes and the odd `.txt`/`.exe` viewer stub that ships on
/// hospital CDs are not images; skipping them by name keeps the "skipped"
/// list free of noise.
pub fn looks_skippable(name: &str) -> bool {
    let lower = name.to_ascii_lowercase();
    // `.xr3d` is a mesh bundle for the 3D view, not an image — but it is very
    // much wanted, so it has to survive this filter.
    if lower.ends_with(".xr3d") {
        return false;
    }
    lower == "dicomdir"
        || [".txt", ".exe", ".dll", ".ini", ".html", ".htm", ".pdf", ".jpg", ".png"]
            .iter()
            .any(|ext| lower.ends_with(ext))
}
