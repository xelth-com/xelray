//! Does the prefetch window actually buy anything?
//!
//! These tests simulate scrolling against a cache with realistic latency —
//! a decode requested at step *t* is not usable until step *t + LATENCY* —
//! and count how often the viewer asks for an image that is not ready yet.
//! Each of those is a visible stutter.
//!
//! The baseline is the symmetric ±3 window this replaced.

use std::collections::HashMap;

use xelray_core::prefetch_order;

/// Steps between issuing a decode and it becoming usable, when the user is
/// stepping slowly enough that a read and decode keep up.
const LATENCY_IDLE: u32 = 2;

/// …and when they are not. Reading half a megabyte off a disc and decoding
/// it takes far longer than one flick of a wheel, so under a brisk scroll
/// several steps pass before a request lands. This is the case the user
/// reported, and the only one where the shape of the window matters.
const LATENCY_BUSY: u32 = 5;

/// The old behaviour: three either side, regardless of travel.
fn symmetric(current: usize, count: usize) -> Vec<usize> {
    let mut out = Vec::new();
    for d in 1..=3i64 {
        for n in [current as i64 - d, current as i64 + d] {
            if n >= 0 && n < count as i64 {
                out.push(n as usize);
            }
        }
    }
    out
}

/// Walk `path`, counting the steps where the wanted image was not ready.
///
/// `plan` is given the current index and the direction of travel.
fn stutters(
    path: &[usize],
    latency: u32,
    plan: impl Fn(usize, i32) -> Vec<usize>,
) -> usize {
    // index -> step at which it becomes usable
    let mut ready: HashMap<usize, u32> = HashMap::new();
    let mut misses = 0;

    for (step, &current) in path.iter().enumerate() {
        let step = step as u32;
        let direction = if step == 0 {
            1
        } else {
            (current as i64 - path[step as usize - 1] as i64).signum() as i32
        };

        // The image on screen: a miss if it was never asked for, or asked
        // for too recently to have finished.
        match ready.get(&current) {
            Some(&t) if t <= step => {}
            _ => {
                misses += 1;
                // A miss is still requested, and lands `latency` later.
                ready.entry(current).or_insert(step + latency);
            }
        }

        for want in plan(current, direction) {
            ready.entry(want).or_insert(step + latency);
        }
    }
    misses
}

/// A long run in one direction, then a reversal — the pattern the user
/// reported as stuttery.
fn forward_then_back() -> Vec<usize> {
    let mut path: Vec<usize> = (0..60).collect();
    path.extend((0..60).rev());
    path
}

/// Ambling one image at a time, decodes keeping up.
///
/// Recorded rather than celebrated: at this pace ±3 was already enough, and
/// the new window must simply not regress it. The bug was never here.
#[test]
fn a_slow_walk_was_never_the_problem() {
    let path = forward_then_back();
    let count = 60;

    let old = stutters(&path, LATENCY_IDLE, |i, _| symmetric(i, count));
    let new = stutters(&path, LATENCY_IDLE, |i, d| prefetch_order(i, count, d, false));

    assert!(new <= old, "regressed the easy case: {new} vs {old}");
    println!("slow walk — symmetric: {old}, directional: {new}");
}

/// The reported symptom: a brisk scroll, reversed halfway.
///
/// Under real decode latency the symmetric window is spending half its
/// budget behind the cursor — on images the cache already holds — and so
/// runs out of runway ahead. Weighting towards travel is what fixes it.
#[test]
fn directional_prefetch_beats_a_symmetric_window_under_load() {
    // Two images per step, decodes five steps behind: a wheel being turned.
    let mut path: Vec<usize> = (0..40).map(|i| i * 2).collect();
    path.extend((0..40).rev().map(|i| i * 2));
    let count = 120;

    let old = stutters(&path, LATENCY_BUSY, |i, _| symmetric(i, count));
    let new = stutters(&path, LATENCY_BUSY, |i, d| prefetch_order(i, count, d, false));
    let fast = stutters(&path, LATENCY_BUSY, |i, d| prefetch_order(i, count, d, true));

    println!("brisk scroll — symmetric: {old}, directional: {new}, fast window: {fast}");
    assert!(
        new < old,
        "directional prefetch should stutter less: {new} vs {old}"
    );
    // Widening the window once the user is clearly moving should help again.
    assert!(
        fast <= new,
        "the fast window should not be worse: {fast} vs {new}"
    );
}

#[test]
fn the_window_flips_the_moment_direction_does() {
    let ahead = prefetch_order(50, 100, 1, false);
    let behind = prefetch_order(50, 100, -1, false);

    assert_eq!(ahead.first(), Some(&51));
    assert_eq!(behind.first(), Some(&49));
    // Mirror images: same shape, opposite travel.
    assert_eq!(ahead.len(), behind.len());
    assert!(ahead[..xelray_core::PREFETCH_AHEAD].iter().all(|&i| i > 50));
    assert!(behind[..xelray_core::PREFETCH_AHEAD].iter().all(|&i| i < 50));
}

#[test]
fn nearest_images_are_requested_first() {
    let order = prefetch_order(50, 100, 1, false);
    let ahead: Vec<usize> = order[..xelray_core::PREFETCH_AHEAD].to_vec();
    assert_eq!(ahead, vec![51, 52, 53, 54, 55, 56]);
    // …then the ones behind, also nearest-first.
    assert_eq!(&order[xelray_core::PREFETCH_AHEAD..], &[49, 48]);
}

#[test]
fn the_window_is_clipped_at_both_ends() {
    // At the very start, nothing behind exists.
    let at_start = prefetch_order(0, 10, 1, false);
    assert!(at_start.iter().all(|&i| i < 10));
    assert!(!at_start.contains(&0));

    // At the very end, nothing ahead does.
    let at_end = prefetch_order(9, 10, 1, false);
    assert!(at_end.iter().all(|&i| i < 10));
    assert!(!at_end.contains(&9));

    // A one-image series has nothing to prefetch at all.
    assert!(prefetch_order(0, 1, 1, false).is_empty());
    assert!(prefetch_order(0, 0, 1, false).is_empty());
    // An out-of-range cursor must not panic or invent indices.
    assert!(prefetch_order(99, 10, 1, false).is_empty());
}

#[test]
fn no_index_is_requested_twice() {
    for fast in [false, true] {
        for direction in [-1, 1] {
            let order = prefetch_order(50, 100, direction, fast);
            let mut sorted = order.clone();
            sorted.sort_unstable();
            sorted.dedup();
            assert_eq!(sorted.len(), order.len(), "duplicate in {order:?}");
        }
    }
}
