//! Integration tests over real CT slices.
//!
//! The fixtures in `tests/data/` are real patient images and are
//! git-ignored on purpose. When they are absent (a fresh clone, CI) every
//! test here degrades to a no-op rather than failing, so `cargo test` stays
//! green for contributors who do not have the study.

use std::path::PathBuf;

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
    assert_eq!(series.modality, "CT");
    assert!(series.warnings.is_empty(), "{:?}", series.warnings);
    assert!(!study.info.patient_name.is_empty());
}

#[test]
fn slices_are_512_squared_and_decode_to_hounsfield_units() {
    let Some(files) = load() else { return };
    let study = xelray_core::ingest(files);
    let series = &study.series[0];

    for inst in &series.instances {
        assert_eq!(inst.rows(), 512);
        assert_eq!(inst.columns(), 512);

        let slice = inst.decode().expect("uncompressed CT must decode");
        assert_eq!(slice.rows, 512);
        assert_eq!(slice.columns, 512);
        assert_eq!(slice.pixels.len(), 512 * 512);

        // After the Modality LUT, CT values are Hounsfield units: air is
        // about -1000, dense bone a few thousand. A buffer still sitting in
        // raw stored values would start at 0.
        let (lo, hi) = slice.value_range();
        assert!(lo < -500.0, "min {lo} is not air-like");
        assert!(hi > 100.0, "max {hi} is too low for CT");
    }
}

#[test]
fn slices_are_sorted_along_the_normal() {
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

    assert!(series.instances.iter().all(|i| i.position.is_some()));
    assert!(series.instances[0].pixel_spacing().is_some());
}

#[test]
fn default_window_comes_from_the_tags() {
    let Some(files) = load() else { return };
    let study = xelray_core::ingest(files);
    let (width, _center) = study.series[0]
        .default_window()
        .expect("CT carries WindowWidth/WindowCenter");
    assert!(width > 0.0);
}

#[test]
fn non_dicom_input_is_reported_not_panicked() {
    let study = xelray_core::ingest(vec![
        ("readme.txt".into(), b"hello".to_vec()),
        ("empty.bin".into(), Vec::new()),
    ]);
    assert!(study.series.is_empty());
    assert_eq!(study.skipped.len(), 2);
}
