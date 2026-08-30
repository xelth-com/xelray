//! Integration tests over real CT slices.
//!
//! The fixtures in `tests/data/` are real patient images and are
//! git-ignored on purpose. When they are absent (a fresh clone, CI) every
//! test here degrades to a no-op rather than failing, so `cargo test` stays
//! green for contributors who do not have the study.

use std::path::PathBuf;

use xelray_core::{SliceCache, HEADER_PREFIX_BYTES};

fn data_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/data")
}

/// Load the fixture files, or `None` when the directory is empty.
fn load() -> Option<Vec<(String, Vec<u8>)>> {
    let dir = data_dir();
    let mut files: Vec<(String, Vec<u8>)> = std::fs::read_dir(&dir)
        .ok()?
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "dcm"))
        .map(|e| {
            let name = e.file_name().to_string_lossy().into_owned();
            (name, std::fs::read(e.path()).expect("fixture unreadable"))
        })
        .collect();

    if files.is_empty() {
        eprintln!("skipping: no fixtures in {}", dir.display());
        return None;
    }
    // Deliberately hand them to `ingest` out of order — sorting is what we
    // are testing.
    files.sort_by(|a, b| b.0.cmp(&a.0));
    Some(files)
}

#[test]
fn ingests_a_ct_study_into_one_series() {
    let Some(files) = load() else { return };
    let n = files.len();
    let study = xelray_core::ingest(files);

    assert!(study.skipped.is_empty(), "skipped files: {:?}", study.skipped);
    assert_eq!(study.series.len(), 1, "fixtures are all one CT series");

    let series = &study.series[0];
    assert_eq!(series.len(), n);
    assert_eq!(study.image_count(), n);
    assert_eq!(series.modality, "CT");
    assert!(series.warnings.is_empty(), "{:?}", series.warnings);
    assert!(!study.info.patient_name.is_empty());
}

/// The whole memory strategy rests on this: a 64 KiB prefix is enough to
/// index a slice, so a study is never read into memory in full.
#[test]
fn a_64k_prefix_is_enough_to_index_a_slice() {
    let Some(files) = load() else { return };
    let (name, bytes) = &files[0];
    assert!(
        bytes.len() > HEADER_PREFIX_BYTES,
        "fixture is smaller than the prefix; this test would prove nothing"
    );

    let prefix = &bytes[..HEADER_PREFIX_BYTES];
    let from_prefix = xelray_core::parse_header(0, name, prefix).expect("prefix must parse");
    let from_whole = xelray_core::parse_header(0, name, bytes).expect("whole file must parse");

    assert_eq!(from_prefix.rows, 512);
    assert_eq!(from_prefix.columns, 512);
    // The prefix must yield exactly what the full read does.
    assert_eq!(from_prefix.rows, from_whole.rows);
    assert_eq!(from_prefix.instance_number, from_whole.instance_number);
    assert_eq!(from_prefix.position, from_whole.position);
    assert_eq!(from_prefix.series_instance_uid, from_whole.series_instance_uid);
    assert_eq!(from_prefix.window, from_whole.window);
    assert_eq!(from_prefix.transfer_syntax, "1.2.840.10008.1.2.1");
}

/// A prefix too short to reach the end of the header must ask for more bytes
/// rather than report the file as junk — that distinction is what stops the
/// loader from silently dropping images.
#[test]
fn a_short_prefix_reports_incomplete_not_invalid() {
    let Some(files) = load() else { return };
    let (name, bytes) = &files[0];

    let err = xelray_core::parse_header(0, name, &bytes[..400])
        .expect_err("400 bytes cannot hold a whole header");
    assert!(err.is_incomplete(), "wrong error kind: {err}");

    // …and the retry with everything succeeds.
    assert!(xelray_core::parse_header(0, name, bytes).is_ok());
}

#[test]
fn slices_are_512_squared_and_decode_to_hounsfield_units() {
    let Some(files) = load() else { return };
    let study = xelray_core::ingest(files.clone());
    let series = &study.series[0];

    for inst in &series.instances {
        assert_eq!(inst.rows, 512);
        assert_eq!(inst.columns, 512);

        let bytes = &files[inst.file_index].1;
        let slice = xelray_core::decode_slice(bytes).expect("uncompressed CT must decode");
        assert_eq!(slice.rows, 512);
        assert_eq!(slice.columns, 512);
        assert_eq!(slice.pixels.len(), 512 * 512);
        assert_eq!(slice.byte_len(), 512 * 512 * 4);

        // After the Modality LUT, CT values are Hounsfield units: air is
        // about -1000, dense bone a few thousand. A buffer still sitting in
        // raw stored values would start at 0.
        let (lo, hi) = slice.value_range();
        assert!(lo < -500.0, "min {lo} is not air-like");
        assert!(hi > 100.0, "max {hi} is too low for CT");
    }
}

#[test]
fn slices_are_sorted_along_the_normal_and_ids_follow() {
    let Some(files) = load() else { return };
    let study = xelray_core::ingest(files);
    let series = &study.series[0];

    let keys: Vec<f64> = series.instances.iter().map(|i| i.along_normal).collect();
    assert!(
        keys.windows(2).all(|w| w[0] <= w[1]),
        "not monotonic: {keys:?}"
    );

    // Geometry and InstanceNumber should agree (up to direction) on a plain
    // axial CT, so the ordering must be strictly monotonic in both.
    let nums: Vec<i32> = series.instances.iter().map(|i| i.instance_number).collect();
    let ascending = nums.windows(2).all(|w| w[0] < w[1]);
    let descending = nums.windows(2).all(|w| w[0] > w[1]);
    assert!(ascending || descending, "instance numbers jumbled: {nums:?}");

    // Cache ids are dense and follow display order.
    let ids: Vec<usize> = series.instances.iter().map(|i| i.id).collect();
    assert_eq!(ids, (0..ids.len()).collect::<Vec<_>>());

    assert!(series.instances.iter().all(|i| i.position.is_some()));
    assert!(series.instances[0].pixel_spacing.is_some());
}

#[test]
fn default_window_comes_from_the_tags() {
    let Some(files) = load() else { return };
    let study = xelray_core::ingest(files);
    let (width, _center) = study.series[0]
        .default_window
        .expect("CT carries WindowWidth/WindowCenter");
    assert!(width > 0.0);
}

/// Walking a whole series the way the viewer does — decode, cache, move on —
/// must leave the cache bounded no matter how long the series is.
#[test]
fn walking_a_series_keeps_the_cache_bounded() {
    let Some(files) = load() else { return };
    let study = xelray_core::ingest(files.clone());
    let mut cache = SliceCache::new(2 * 1024 * 1024, 96);

    // Two megabytes holds two 512² slices; four passes over the fixtures is
    // enough to force eviction and re-decoding.
    for _ in 0..4 {
        for inst in &study.series[0].instances {
            if cache.get(inst.id).is_none() {
                let bytes = &files[inst.file_index].1;
                let slice = xelray_core::decode_slice(bytes).expect("decode");
                cache.insert(inst.id, std::rc::Rc::new(slice));
            }
            assert!(cache.byte_len() <= 2 * 1024 * 1024);
        }
    }
    assert_eq!(cache.len(), 2);
}

#[test]
fn non_dicom_input_is_reported_not_panicked() {
    let study = xelray_core::ingest(vec![
        ("readme.txt".into(), b"this is definitely not a DICOM file at all".to_vec()),
        ("empty.bin".into(), Vec::new()),
    ]);
    assert!(study.series.is_empty());
    assert_eq!(study.skipped.len(), 2);
}

#[test]
fn decoding_junk_fails_without_panicking() {
    assert!(xelray_core::decode_slice(b"not dicom").is_err());
    assert!(xelray_core::decode_slice(&[0u8; 200]).is_err());
}
