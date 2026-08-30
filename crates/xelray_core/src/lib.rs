//! XelRay core — pure-Rust DICOM ingest for the browser.
//!
//! The crate is deliberately free of any wasm-, DOM- or filesystem-specific
//! code: it turns file bytes into headers, headers into a sorted [`Study`],
//! and — separately, on demand — one file's bytes into a decoded [`Slice`].
//! That keeps the whole parsing path unit-testable on the native target
//! while the very same code runs inside `wasm32-unknown-unknown`.
//!
//! # Memory
//!
//! The split between [`parse_header`] and [`decode_slice`] is the load-bearing
//! design decision. A 1000-image CT is half a gigabyte on disk and a gigabyte
//! decoded — both far beyond a 32-bit wasm heap. So nothing here ever holds a
//! whole study: [`parse_header`] reads a bounded prefix of a file, keeps a few
//! hundred bytes of metadata and drops the rest, and [`decode_slice`] is
//! called for one image at a time with its result parked in a bounded
//! [`SliceCache`]. The caller keeps the file *handles*, never their contents.

mod cache;

use std::collections::HashMap;

use dicom_core::value::Value;
use dicom_dictionary_std::tags;
use dicom_object::{FileDicomObject, InMemDicomObject, OpenFileOptions};
use dicom_pixeldata::PixelDecoder;

pub use cache::{SliceCache, DEFAULT_MAX_BYTES, DEFAULT_MAX_SLICES};

/// How much of a file is normally enough to reach the pixel data.
///
/// Everything before `PixelData` is what we index on, and in practice that is
/// a couple of kilobytes. 64 KiB leaves generous room for private tags and an
/// icon-image sequence while still being a fraction of a slice.
pub const HEADER_PREFIX_BYTES: usize = 64 * 1024;

type Obj = FileDicomObject<InMemDicomObject>;

/// Everything [`parse_header`] recovers from one file.
///
/// This is a transient value: [`push_header`] moves the study- and
/// series-level strings out of it into the [`Study`], and only the slim
/// [`Instance`] is retained per image.
#[derive(Debug, Clone)]
pub struct Header {
    pub file_index: usize,
    pub file_name: String,
    pub instance_number: i32,
    pub position: Option<[f64; 3]>,
    pub orientation: Option<[f64; 6]>,
    pub rows: u16,
    pub columns: u16,
    pub pixel_spacing: Option<(f64, f64)>,
    /// `WindowWidth` / `WindowCenter`, when the file states them.
    pub window: Option<(f64, f64)>,
    pub transfer_syntax: String,
    pub series_instance_uid: String,
    pub series_description: String,
    pub series_number: i32,
    pub modality: String,
    pub patient_name: String,
    pub patient_id: String,
    pub study_date: String,
    pub study_description: String,
}

/// One image's place in the study — a few dozen bytes, no pixels.
///
/// A thousand of these is well under a megabyte, which is what makes
/// indexing a whole hospital CD affordable.
pub struct Instance {
    /// Unique within the study; the [`SliceCache`] key.
    pub id: usize,
    /// Index into the caller's own list of file handles.
    pub file_index: usize,
    pub file_name: String,
    pub instance_number: i32,
    pub position: Option<[f64; 3]>,
    /// Projection of `position` onto the slice normal — the sort key.
    pub along_normal: f64,
    pub rows: u16,
    pub columns: u16,
    pub pixel_spacing: Option<(f64, f64)>,
}

/// One decoded slice: a flat row-major buffer of modality values.
pub struct Slice {
    pub rows: usize,
    pub columns: usize,
    /// `rows * columns` values, row-major, already rescaled.
    pub pixels: Vec<f32>,
}

impl Slice {
    /// Bytes of heap this slice occupies, for the cache's budget.
    pub fn byte_len(&self) -> usize {
        self.pixels.len() * std::mem::size_of::<f32>()
    }

    /// Min/max of the buffer — the fallback window when the file carries no
    /// `WindowCenter`/`WindowWidth`.
    pub fn value_range(&self) -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &v in &self.pixels {
            if v < lo {
                lo = v;
            }
            if v > hi {
                hi = v;
            }
        }
        if lo > hi {
            (0.0, 1.0)
        } else {
            (lo, hi)
        }
    }
}

/// A series: the unit the viewer scrolls through.
pub struct Series {
    pub series_instance_uid: String,
    pub series_description: String,
    pub series_number: i32,
    pub modality: String,
    /// Slices sorted head-to-foot (or by `InstanceNumber` where geometry is
    /// missing).
    pub instances: Vec<Instance>,
    /// `WindowWidth`/`WindowCenter` from the first image seen.
    pub default_window: Option<(f64, f64)>,
    /// Direction cosines from the first image, used for sorting.
    orientation: Option<[f64; 6]>,
    /// Non-fatal problems — e.g. a transfer syntax this build cannot decode.
    /// Surfaced in the UI as a per-series warning instead of a panic.
    pub warnings: Vec<String>,
}

impl Series {
    pub fn len(&self) -> usize {
        self.instances.len()
    }

    pub fn is_empty(&self) -> bool {
        self.instances.is_empty()
    }

    /// A human label for the series list.
    pub fn label(&self) -> String {
        if self.series_description.is_empty() {
            format!("{} series {}", self.modality, self.series_number)
        } else {
            self.series_description.clone()
        }
    }
}

/// Patient/study level metadata, taken from the first instance seen.
#[derive(Default, Clone)]
pub struct StudyInfo {
    pub patient_name: String,
    pub patient_id: String,
    pub study_date: String,
    pub study_description: String,
}

/// The index of a loaded folder: metadata only, never pixels.
#[derive(Default)]
pub struct Study {
    pub info: StudyInfo,
    pub series: Vec<Series>,
    /// Files that were skipped (not DICOM, truncated, unreadable).
    pub skipped: Vec<(String, String)>,
    /// Running id counter, handed out by [`finalize`].
    next_id: usize,
}

impl Study {
    /// Total images indexed across every series.
    pub fn image_count(&self) -> usize {
        self.series.iter().map(|s| s.len()).sum()
    }
}

/// Errors that abort a single file or a single decode — never the whole load.
#[derive(Debug)]
pub enum XelRayError {
    /// Not a DICOM file at all — no `DICM` magic. Retrying with more bytes
    /// would not help.
    NotDicom(String),
    /// The bytes given were a valid DICOM prefix but ran out before the
    /// header was complete. The caller should retry with the whole file.
    Incomplete(String),
    /// Parsed, but the pixel data could not be decoded in this build.
    Decode(String),
}

impl XelRayError {
    /// Whether re-reading the file in full might succeed.
    pub fn is_incomplete(&self) -> bool {
        matches!(self, XelRayError::Incomplete(_))
    }
}

impl std::fmt::Display for XelRayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            XelRayError::NotDicom(m) => write!(f, "not a DICOM file: {m}"),
            XelRayError::Incomplete(m) => write!(f, "incomplete DICOM header: {m}"),
            XelRayError::Decode(m) => write!(f, "cannot decode pixel data: {m}"),
        }
    }
}

impl std::error::Error for XelRayError {}

/// Transfer syntaxes this build knowingly cannot decode.
///
/// JPEG 2000 needs OpenJPEG, a C library that does not build for
/// `wasm32-unknown-unknown`; rather than let every slice fail with an opaque
/// message we detect it up front and warn once per series.
fn unsupported_reason(ts_uid: &str) -> Option<&'static str> {
    match ts_uid.trim_end_matches('\0') {
        "1.2.840.10008.1.2.4.90" | "1.2.840.10008.1.2.4.91" => {
            Some("JPEG 2000 — not supported in the WebAssembly build")
        }
        "1.2.840.10008.1.2.4.92" | "1.2.840.10008.1.2.4.93" => {
            Some("JPEG 2000 Part 2 — not supported in the WebAssembly build")
        }
        "1.2.840.10008.1.2.4.100" | "1.2.840.10008.1.2.4.101" | "1.2.840.10008.1.2.4.102" => {
            Some("MPEG video — not supported")
        }
        "1.2.840.10008.1.2.4.110" | "1.2.840.10008.1.2.4.111" | "1.2.840.10008.1.2.4.112"
        | "1.2.840.10008.1.2.4.113" => Some("JPEG XL — not supported in the WebAssembly build"),
        _ => None,
    }
}

/// Strip the part-10 preamble, leaving the slice at the `DICM` magic.
///
/// `dicom-object`'s reader wants to read the magic itself, so the returned
/// slice starts *at* it, not after it. Some archives omit the 128-byte
/// preamble entirely, so both layouts are accepted.
fn body_at_magic(bytes: &[u8]) -> Result<&[u8], XelRayError> {
    if bytes.len() >= 132 && &bytes[128..132] == b"DICM" {
        Ok(&bytes[128..])
    } else if bytes.len() >= 4 && &bytes[0..4] == b"DICM" {
        Ok(bytes)
    } else if bytes.len() < 132 {
        // Too short to tell — could be a truncated read of a real file.
        Err(XelRayError::Incomplete("fewer than 132 bytes".into()))
    } else {
        Err(XelRayError::NotDicom("missing DICM magic".into()))
    }
}

/// Read one file's metadata, stopping at the pixel data.
///
/// `bytes` may be a prefix of the file — [`HEADER_PREFIX_BYTES`] is enough
/// for essentially every real image. If the header turns out to run past the
/// end of the prefix the result is [`XelRayError::Incomplete`], and the
/// caller should retry with the whole file.
///
/// Because the read stops before `PixelData`, the megabyte of pixels is never
/// allocated: indexing a study costs its metadata, not its size.
pub fn parse_header(
    file_index: usize,
    file_name: &str,
    bytes: &[u8],
) -> Result<Header, XelRayError> {
    let body = body_at_magic(bytes)?;

    let obj = OpenFileOptions::new()
        .read_until(tags::PIXEL_DATA)
        .from_reader(body)
        // Any failure past the magic is treated as "not enough bytes". A file
        // that is genuinely corrupt will fail the same way on the full read
        // and be reported then, which costs one wasted read on a broken file
        // and saves guessing here.
        .map_err(|e| XelRayError::Incomplete(e.to_string()))?;

    let ipp = get_f64_multi(&obj, tags::IMAGE_POSITION_PATIENT);
    let position = match ipp.as_slice() {
        [x, y, z, ..] => Some([*x, *y, *z]),
        _ => None,
    };

    let iop = get_f64_multi(&obj, tags::IMAGE_ORIENTATION_PATIENT);
    let orientation = <[f64; 6]>::try_from(&iop[..6.min(iop.len())]).ok();

    let spacing = get_f64_multi(&obj, tags::PIXEL_SPACING);
    let pixel_spacing = match spacing.as_slice() {
        [r, c, ..] => Some((*r, *c)),
        _ => None,
    };

    let width = get_f64_multi(&obj, tags::WINDOW_WIDTH).into_iter().next();
    let center = get_f64_multi(&obj, tags::WINDOW_CENTER).into_iter().next();

    Ok(Header {
        file_index,
        file_name: file_name.to_owned(),
        instance_number: get_str(&obj, tags::INSTANCE_NUMBER)
            .and_then(|s| s.trim().parse::<i32>().ok())
            .unwrap_or(0),
        position,
        orientation,
        rows: get_u16(&obj, tags::ROWS).unwrap_or(0),
        columns: get_u16(&obj, tags::COLUMNS).unwrap_or(0),
        pixel_spacing,
        window: width.zip(center),
        transfer_syntax: obj.meta().transfer_syntax().to_owned(),
        series_instance_uid: get_str(&obj, tags::SERIES_INSTANCE_UID).unwrap_or_default(),
        series_description: get_str(&obj, tags::SERIES_DESCRIPTION).unwrap_or_default(),
        series_number: get_str(&obj, tags::SERIES_NUMBER)
            .and_then(|s| s.trim().parse::<i32>().ok())
            .unwrap_or(0),
        modality: get_str(&obj, tags::MODALITY).unwrap_or_default(),
        patient_name: get_person_name(&obj, tags::PATIENT_NAME),
        patient_id: get_str(&obj, tags::PATIENT_ID).unwrap_or_default(),
        study_date: format_date(&get_str(&obj, tags::STUDY_DATE).unwrap_or_default()),
        study_description: get_str(&obj, tags::STUDY_DESCRIPTION).unwrap_or_default(),
    })
}

/// Decode one image's pixels into modality (rescaled) values.
///
/// `bytes` must be the *whole* file. The parsed object and the encoded pixel
/// data are both dropped before this returns, so the only lasting cost is the
/// [`Slice`] itself.
///
/// `RescaleSlope`/`RescaleIntercept` are applied by `dicom-pixeldata`'s
/// default Modality-LUT pipeline, so for CT the result is Hounsfield units
/// and the window presets are directly meaningful.
pub fn decode_slice(bytes: &[u8]) -> Result<Slice, XelRayError> {
    let body = body_at_magic(bytes)?;
    let obj = FileDicomObject::from_reader(body).map_err(|e| XelRayError::Decode(e.to_string()))?;

    if let Some(reason) = unsupported_reason(obj.meta().transfer_syntax()) {
        return Err(XelRayError::Decode(reason.to_owned()));
    }

    let decoded = obj
        .decode_pixel_data_frame(0)
        .map_err(|e| XelRayError::Decode(e.to_string()))?;

    let rows = decoded.rows() as usize;
    let columns = decoded.columns() as usize;
    let pixels: Vec<f32> = decoded
        .to_vec_frame(0)
        .map_err(|e| XelRayError::Decode(e.to_string()))?;

    // A malformed file could otherwise hand the renderer a buffer that does
    // not match its own stated geometry.
    if pixels.len() < rows * columns {
        return Err(XelRayError::Decode(format!(
            "pixel buffer holds {} values, expected {}",
            pixels.len(),
            rows * columns
        )));
    }

    Ok(Slice {
        rows,
        columns,
        pixels,
    })
}

/// Add one parsed header to a study under construction.
///
/// The caller drives this file by file, yielding to the event loop in
/// between, and calls [`finalize`] once at the end.
pub fn push_header(study: &mut Study, by_uid: &mut HashMap<String, usize>, header: Header) {
    if study.info.patient_name.is_empty() {
        study.info = StudyInfo {
            patient_name: header.patient_name.clone(),
            patient_id: header.patient_id.clone(),
            study_date: header.study_date.clone(),
            study_description: header.study_description.clone(),
        };
    }

    let idx = *by_uid
        .entry(header.series_instance_uid.clone())
        .or_insert_with(|| {
            let mut warnings = Vec::new();
            if let Some(reason) = unsupported_reason(&header.transfer_syntax) {
                warnings.push(format!("Compressed with {reason}. Images cannot be shown."));
            }
            study.series.push(Series {
                series_instance_uid: header.series_instance_uid.clone(),
                series_description: header.series_description.clone(),
                series_number: header.series_number,
                modality: header.modality.clone(),
                instances: Vec::new(),
                default_window: header.window,
                orientation: header.orientation,
                warnings,
            });
            study.series.len() - 1
        });

    study.series[idx].instances.push(Instance {
        // Real ids are handed out by `finalize`, once the order is settled.
        id: 0,
        file_index: header.file_index,
        file_name: header.file_name,
        instance_number: header.instance_number,
        position: header.position,
        along_normal: 0.0,
        rows: header.rows,
        columns: header.columns,
        pixel_spacing: header.pixel_spacing,
    });
}

/// Sort every series' slices, sort the series list, and assign cache ids.
///
/// Slices are ordered by their projection onto the slice normal — the normal
/// comes from `ImageOrientationPatient`, so this is the true through-plane
/// axis rather than a blind `z` compare, which would mis-order oblique
/// acquisitions. Series without usable geometry fall back to
/// `InstanceNumber`.
pub fn finalize(study: &mut Study) {
    for series in &mut study.series {
        match series.orientation.map(normal_of) {
            Some(n) => {
                for inst in &mut series.instances {
                    inst.along_normal = inst
                        .position
                        .map(|p| p[0] * n[0] + p[1] * n[1] + p[2] * n[2])
                        .unwrap_or(inst.instance_number as f64);
                }
                series.instances.sort_by(|a, b| {
                    a.along_normal
                        .partial_cmp(&b.along_normal)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.instance_number.cmp(&b.instance_number))
                });
            }
            None => series.instances.sort_by_key(|i| i.instance_number),
        }
    }

    study
        .series
        .sort_by_key(|s| (s.series_number, s.series_instance_uid.clone()));

    // Ids are assigned last so they are stable for the life of the study and
    // usable directly as cache keys.
    for series in &mut study.series {
        for inst in &mut series.instances {
            inst.id = study.next_id;
            study.next_id += 1;
        }
    }
}

/// Cross product of the two `ImageOrientationPatient` direction cosines.
fn normal_of([rx, ry, rz, cx, cy, cz]: [f64; 6]) -> [f64; 3] {
    [
        ry * cz - rz * cy,
        rz * cx - rx * cz,
        rx * cy - ry * cx,
    ]
}

/// Index a batch of already-read files in one call.
///
/// Convenience for tests and any caller that genuinely has all the bytes to
/// hand; the browser front end drives [`parse_header`] and [`push_header`]
/// itself so it can yield between files and free each one as it goes.
pub fn ingest(files: Vec<(String, Vec<u8>)>) -> Study {
    let mut study = Study::default();
    let mut by_uid: HashMap<String, usize> = HashMap::new();

    for (i, (name, bytes)) in files.iter().enumerate() {
        match parse_header(i, name, bytes) {
            Ok(h) => push_header(&mut study, &mut by_uid, h),
            Err(e) => study.skipped.push((name.clone(), e.to_string())),
        }
    }

    finalize(&mut study);
    study
}

// ---------------------------------------------------------------------------
// Tag readers
//
// dicom-rs returns rich `Value`s; the viewer only ever wants a trimmed string
// or a list of floats, so the conversions live here rather than at every call
// site.
// ---------------------------------------------------------------------------

fn get_str(obj: &Obj, tag: dicom_core::Tag) -> Option<String> {
    let e = obj.element_opt(tag).ok().flatten()?;
    let s = e.to_str().ok()?;
    let s = s.trim().trim_end_matches('\0').trim().to_owned();
    if s.is_empty() {
        None
    } else {
        Some(s)
    }
}

/// `PN` values arrive caret-separated (`Last^First^Middle`); show them the way
/// a human writes a name.
fn get_person_name(obj: &Obj, tag: dicom_core::Tag) -> String {
    let Some(raw) = get_str(obj, tag) else {
        return "(anonymous)".to_owned();
    };
    let parts: Vec<&str> = raw
        .split('^')
        .map(|p| p.trim())
        .filter(|p| !p.is_empty())
        .collect();
    match parts.as_slice() {
        [] => "(anonymous)".to_owned(),
        [last] => (*last).to_owned(),
        [last, rest @ ..] => format!("{} {}", rest.join(" "), last),
    }
}

fn get_u16(obj: &Obj, tag: dicom_core::Tag) -> Option<u16> {
    let e = obj.element_opt(tag).ok().flatten()?;
    e.to_int::<u16>().ok()
}

fn get_f64_multi(obj: &Obj, tag: dicom_core::Tag) -> Vec<f64> {
    let Some(e) = obj.element_opt(tag).ok().flatten() else {
        return Vec::new();
    };
    if let Ok(v) = e.to_multi_float64() {
        return v;
    }
    // Decimal strings occasionally arrive with stray padding that
    // `to_multi_float64` rejects; fall back to a manual split.
    match e.value() {
        Value::Primitive(_) => e
            .to_str()
            .map(|s| {
                s.split('\\')
                    .filter_map(|p| p.trim().trim_end_matches('\0').trim().parse::<f64>().ok())
                    .collect()
            })
            .unwrap_or_default(),
        _ => Vec::new(),
    }
}

/// `YYYYMMDD` → `YYYY-MM-DD`, anything else passes through untouched.
fn format_date(raw: &str) -> String {
    if raw.len() == 8 && raw.bytes().all(|b| b.is_ascii_digit()) {
        format!("{}-{}-{}", &raw[0..4], &raw[4..6], &raw[6..8])
    } else {
        raw.to_owned()
    }
}

/// The window/level presets offered in the toolbar, as `(name, width, center)`.
pub const WINDOW_PRESETS: &[(&str, f64, f64)] = &[
    ("Soft tissue", 400.0, 40.0),
    ("Lung", 1500.0, -600.0),
    ("Bone", 1800.0, 400.0),
    ("Brain", 80.0, 40.0),
];
