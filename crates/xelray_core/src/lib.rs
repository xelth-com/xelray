//! XelRay core — pure-Rust DICOM ingest for the browser.
//!
//! The crate is deliberately free of any wasm-, DOM- or filesystem-specific
//! code: it takes `(name, bytes)` pairs, hands back a [`Study`] of sorted
//! [`Series`], and decodes a slice's pixels on demand. That keeps the whole
//! parsing path unit-testable on the native target while the very same code
//! runs inside `wasm32-unknown-unknown`.

use std::collections::HashMap;

use dicom_core::value::Value;
use dicom_dictionary_std::tags;
use dicom_object::{FileDicomObject, InMemDicomObject};
use dicom_pixeldata::PixelDecoder;

/// A parsed-but-not-yet-decoded DICOM instance.
///
/// The raw bytes are kept around so that pixel decoding stays lazy: a 500
/// slice CT is ~250 MB of pixels, and only the slices actually looked at are
/// ever expanded.
pub struct Instance {
    /// Original file name, kept for diagnostics only.
    pub file_name: String,
    /// Parsed object (header + pixel data element, undecoded).
    obj: FileDicomObject<InMemDicomObject>,
    /// `InstanceNumber` (0020,0013), used as the sort fallback.
    pub instance_number: i32,
    /// `ImagePositionPatient` (0020,0032).
    pub position: Option<[f64; 3]>,
    /// Projection of `position` onto the slice normal — the sort key.
    pub along_normal: f64,
}

impl Instance {
    /// Rows in the image (`0028,0010`).
    pub fn rows(&self) -> u16 {
        get_u16(&self.obj, tags::ROWS).unwrap_or(0)
    }

    /// Columns in the image (`0028,0011`).
    pub fn columns(&self) -> u16 {
        get_u16(&self.obj, tags::COLUMNS).unwrap_or(0)
    }

    /// `PixelSpacing` (0028,0030) as `(row_mm, col_mm)`.
    pub fn pixel_spacing(&self) -> Option<(f64, f64)> {
        let v = get_f64_multi(&self.obj, tags::PIXEL_SPACING);
        match v.as_slice() {
            [r, c, ..] => Some((*r, *c)),
            _ => None,
        }
    }

    /// Decode this slice's pixels into modality (rescaled) values.
    ///
    /// `RescaleSlope`/`RescaleIntercept` are applied by `dicom-pixeldata`'s
    /// default Modality-LUT pipeline, so for CT the result is Hounsfield
    /// units and window presets are directly meaningful.
    pub fn decode(&self) -> Result<Slice, XelRayError> {
        let decoded = self
            .obj
            .decode_pixel_data_frame(0)
            .map_err(|e| XelRayError::Decode(e.to_string()))?;

        let pixels: Vec<f32> = decoded
            .to_vec_frame(0)
            .map_err(|e| XelRayError::Decode(e.to_string()))?;

        Ok(Slice {
            rows: decoded.rows() as usize,
            columns: decoded.columns() as usize,
            pixels,
        })
    }
}

/// One decoded slice: a flat row-major buffer of modality values.
pub struct Slice {
    pub rows: usize,
    pub columns: usize,
    /// `rows * columns` values, row-major, already rescaled.
    pub pixels: Vec<f32>,
}

impl Slice {
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

    /// Default window from the first slice's `WindowWidth`/`WindowCenter`.
    pub fn default_window(&self) -> Option<(f64, f64)> {
        let obj = &self.instances.first()?.obj;
        let width = get_f64_multi(obj, tags::WINDOW_WIDTH).into_iter().next()?;
        let center = get_f64_multi(obj, tags::WINDOW_CENTER).into_iter().next()?;
        Some((width, center))
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

/// The result of ingesting a folder of files.
#[derive(Default)]
pub struct Study {
    pub info: StudyInfo,
    pub series: Vec<Series>,
    /// Files that were skipped (not DICOM, truncated, unreadable).
    pub skipped: Vec<(String, String)>,
}

/// Errors that abort a single file or a single decode — never the whole load.
#[derive(Debug)]
pub enum XelRayError {
    /// Not a DICOM file at all (no `DICM` magic, or the header would not parse).
    NotDicom(String),
    /// Parsed, but the pixel data could not be decoded in this build.
    Decode(String),
}

impl std::fmt::Display for XelRayError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            XelRayError::NotDicom(m) => write!(f, "not a DICOM file: {m}"),
            XelRayError::Decode(m) => write!(f, "cannot decode pixel data: {m}"),
        }
    }
}

impl std::error::Error for XelRayError {}

/// Transfer syntaxes this build knowingly cannot decode.
///
/// JPEG 2000 needs OpenJPEG, a C library that does not build for
/// `wasm32-unknown-unknown`; rather than let `decode_pixel_data` fail with an
/// opaque message per slice we detect it up front and warn once per series.
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

/// Parse one file's bytes into an [`Instance`].
///
/// Handles both plain part-10 files (128-byte preamble + `DICM`) and the
/// preamble-less form some archives produce. `from_reader` wants to read the
/// magic code itself, so the slice we hand it starts *at* `DICM`.
pub fn parse_instance(file_name: &str, bytes: &[u8]) -> Result<Instance, XelRayError> {
    let body = if bytes.len() > 132 && &bytes[128..132] == b"DICM" {
        &bytes[128..]
    } else if bytes.len() > 4 && &bytes[0..4] == b"DICM" {
        bytes
    } else {
        return Err(XelRayError::NotDicom("missing DICM magic".into()));
    };

    let obj = FileDicomObject::from_reader(body).map_err(|e| XelRayError::NotDicom(e.to_string()))?;

    let instance_number = get_str(&obj, tags::INSTANCE_NUMBER)
        .and_then(|s| s.trim().parse::<i32>().ok())
        .unwrap_or(0);

    let ipp = get_f64_multi(&obj, tags::IMAGE_POSITION_PATIENT);
    let position = match ipp.as_slice() {
        [x, y, z, ..] => Some([*x, *y, *z]),
        _ => None,
    };

    Ok(Instance {
        file_name: file_name.to_owned(),
        obj,
        instance_number,
        position,
        along_normal: 0.0,
    })
}

/// Ingest a batch of files into a [`Study`].
///
/// Order of the input does not matter; instances are grouped by
/// `SeriesInstanceUID` and each series is sorted afterwards.
pub fn ingest(files: Vec<(String, Vec<u8>)>) -> Study {
    let mut study = Study::default();
    let mut by_uid: HashMap<String, usize> = HashMap::new();

    for (name, bytes) in files {
        match parse_instance(&name, &bytes) {
            Ok(inst) => push_instance(&mut study, &mut by_uid, inst),
            Err(e) => study.skipped.push((name, e.to_string())),
        }
    }

    finalize(&mut study);
    study
}

/// Add one already-parsed instance to a study under construction.
///
/// Exposed separately so the UI can stream files in chunks — parse a handful,
/// yield to the browser event loop, repeat — and still call [`finalize`] once
/// at the end.
pub fn push_instance(
    study: &mut Study,
    by_uid: &mut HashMap<String, usize>,
    inst: Instance,
) {
    if study.info.patient_name.is_empty() {
        study.info = StudyInfo {
            patient_name: get_person_name(&inst.obj, tags::PATIENT_NAME),
            patient_id: get_str(&inst.obj, tags::PATIENT_ID).unwrap_or_default(),
            study_date: format_date(&get_str(&inst.obj, tags::STUDY_DATE).unwrap_or_default()),
            study_description: get_str(&inst.obj, tags::STUDY_DESCRIPTION).unwrap_or_default(),
        };
    }

    let uid = get_str(&inst.obj, tags::SERIES_INSTANCE_UID).unwrap_or_default();
    let idx = *by_uid.entry(uid.clone()).or_insert_with(|| {
        let mut warnings = Vec::new();
        let ts = inst.obj.meta().transfer_syntax();
        if let Some(reason) = unsupported_reason(ts) {
            warnings.push(format!("Compressed with {reason}. Images cannot be shown."));
        }
        study.series.push(Series {
            series_instance_uid: uid.clone(),
            series_description: get_str(&inst.obj, tags::SERIES_DESCRIPTION).unwrap_or_default(),
            series_number: get_str(&inst.obj, tags::SERIES_NUMBER)
                .and_then(|s| s.trim().parse::<i32>().ok())
                .unwrap_or(0),
            modality: get_str(&inst.obj, tags::MODALITY).unwrap_or_default(),
            instances: Vec::new(),
            warnings,
        });
        study.series.len() - 1
    });

    study.series[idx].instances.push(inst);
}

/// Sort every series' slices and the series list itself.
///
/// Slices are ordered by their projection onto the slice normal — the normal
/// comes from `ImageOrientationPatient`, so this is the true through-plane
/// axis rather than a blind `z` compare, which would mis-order oblique
/// acquisitions. Series without usable geometry fall back to
/// `InstanceNumber`.
pub fn finalize(study: &mut Study) {
    for series in &mut study.series {
        let normal = series
            .instances
            .first()
            .and_then(|i| slice_normal(&i.obj));

        match normal {
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
}

/// Cross product of the two `ImageOrientationPatient` direction cosines.
fn slice_normal(obj: &FileDicomObject<InMemDicomObject>) -> Option<[f64; 3]> {
    let v = get_f64_multi(obj, tags::IMAGE_ORIENTATION_PATIENT);
    let [rx, ry, rz, cx, cy, cz] = <[f64; 6]>::try_from(&v[..6.min(v.len())]).ok()?;
    Some([
        ry * cz - rz * cy,
        rz * cx - rx * cz,
        rx * cy - ry * cx,
    ])
}

// ---------------------------------------------------------------------------
// Tag readers
//
// dicom-rs returns rich `Value`s; the viewer only ever wants a trimmed string
// or a list of floats, so the conversions live here rather than at every call
// site.
// ---------------------------------------------------------------------------

fn get_str(obj: &FileDicomObject<InMemDicomObject>, tag: dicom_core::Tag) -> Option<String> {
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
fn get_person_name(obj: &FileDicomObject<InMemDicomObject>, tag: dicom_core::Tag) -> String {
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

fn get_u16(obj: &FileDicomObject<InMemDicomObject>, tag: dicom_core::Tag) -> Option<u16> {
    let e = obj.element_opt(tag).ok().flatten()?;
    e.to_int::<u16>().ok()
}

fn get_f64_multi(obj: &FileDicomObject<InMemDicomObject>, tag: dicom_core::Tag) -> Vec<f64> {
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
