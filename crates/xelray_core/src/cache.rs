//! A byte-budgeted LRU for decoded slices.
//!
//! Decoded pixels dominate this application's memory: one 512² CT slice is a
//! megabyte as `f32`, and a study has a thousand of them. Holding even a
//! fraction of that alongside the wasm heap's other tenants is what makes a
//! tab die, so the viewer keeps a strictly bounded window of recent slices
//! and re-decodes anything that falls out.
//!
//! The budget is in bytes rather than entries because slice size varies by an
//! order of magnitude between a 256² ultrasound frame and a 1024² MR.

use std::collections::{HashMap, VecDeque};
use std::rc::Rc;

use crate::Slice;

/// Default ceiling for decoded pixels: comfortably inside a wasm32 heap even
/// with a large study indexed alongside it.
pub const DEFAULT_MAX_BYTES: usize = 48 * 1024 * 1024;

/// Hard cap on entries, so a stream of tiny images cannot make the map huge.
pub const DEFAULT_MAX_SLICES: usize = 96;

/// Least-recently-used cache keyed by [`crate::Instance::id`].
pub struct SliceCache {
    max_bytes: usize,
    max_slices: usize,
    bytes: usize,
    entries: HashMap<usize, Rc<Slice>>,
    /// Front is the least recently used.
    order: VecDeque<usize>,
}

impl Default for SliceCache {
    fn default() -> Self {
        Self::new(DEFAULT_MAX_BYTES, DEFAULT_MAX_SLICES)
    }
}

impl SliceCache {
    pub fn new(max_bytes: usize, max_slices: usize) -> Self {
        Self {
            max_bytes,
            max_slices: max_slices.max(1),
            bytes: 0,
            entries: HashMap::new(),
            order: VecDeque::new(),
        }
    }

    /// Fetch a slice, marking it as most recently used.
    pub fn get(&mut self, id: usize) -> Option<Rc<Slice>> {
        let slice = self.entries.get(&id)?.clone();
        self.touch(id);
        Some(slice)
    }

    /// True if the slice is resident, without disturbing the LRU order.
    ///
    /// Prefetch uses this: promoting a speculatively fetched neighbour ahead
    /// of the slice actually on screen would be exactly backwards.
    pub fn contains(&self, id: usize) -> bool {
        self.entries.contains_key(&id)
    }

    /// Insert a slice, evicting until the cache is back inside its budget.
    ///
    /// An entry larger than the whole budget is still stored — it is what the
    /// user asked to look at — but it will be alone.
    pub fn insert(&mut self, id: usize, slice: Rc<Slice>) {
        if self.entries.contains_key(&id) {
            self.touch(id);
            return;
        }
        self.bytes += slice.byte_len();
        self.entries.insert(id, slice);
        self.order.push_back(id);

        while self.order.len() > 1
            && (self.bytes > self.max_bytes || self.order.len() > self.max_slices)
        {
            let Some(oldest) = self.order.pop_front() else {
                break;
            };
            if let Some(dropped) = self.entries.remove(&oldest) {
                self.bytes = self.bytes.saturating_sub(dropped.byte_len());
            }
        }
    }

    /// Forget everything — used when a new study is loaded.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.order.clear();
        self.bytes = 0;
    }

    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Bytes of decoded pixel data currently held.
    pub fn byte_len(&self) -> usize {
        self.bytes
    }

    fn touch(&mut self, id: usize) {
        if let Some(pos) = self.order.iter().position(|&x| x == id) {
            self.order.remove(pos);
            self.order.push_back(id);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn slice_of(px: usize) -> Rc<Slice> {
        Rc::new(Slice {
            rows: px,
            columns: px,
            pixels: vec![0.0; px * px],
        })
    }

    /// The point of the whole exercise: streaming a study far larger than
    /// memory through the cache must not grow it.
    #[test]
    fn six_hundred_slices_stay_inside_the_budget() {
        let mut cache = SliceCache::new(32 * 1024 * 1024, 96);
        for id in 0..600 {
            cache.insert(id, slice_of(512));
            assert!(
                cache.byte_len() <= 32 * 1024 * 1024,
                "budget blown at {id}: {} bytes",
                cache.byte_len()
            );
            assert!(cache.len() <= 96);
        }
        // 1 MiB per slice against a 32 MiB budget.
        assert_eq!(cache.len(), 32);
        assert!(cache.contains(599), "the newest slice must survive");
        assert!(!cache.contains(0), "the oldest must have been evicted");
    }

    #[test]
    fn entry_cap_binds_when_slices_are_small() {
        let mut cache = SliceCache::new(usize::MAX, 8);
        for id in 0..50 {
            cache.insert(id, slice_of(16));
        }
        assert_eq!(cache.len(), 8);
    }

    #[test]
    fn reading_a_slice_protects_it_from_eviction() {
        let mut cache = SliceCache::new(4 * 1024 * 1024, 96);
        for id in 0..4 {
            cache.insert(id, slice_of(512));
        }
        // Slice 0 would be evicted next; reading it moves it to the back.
        assert!(cache.get(0).is_some());
        cache.insert(4, slice_of(512));

        assert!(cache.contains(0), "recently read slice was evicted");
        assert!(!cache.contains(1), "slice 1 should have gone instead");
    }

    #[test]
    fn contains_does_not_reorder() {
        let mut cache = SliceCache::new(4 * 1024 * 1024, 96);
        for id in 0..4 {
            cache.insert(id, slice_of(512));
        }
        assert!(cache.contains(0));
        cache.insert(4, slice_of(512));
        assert!(!cache.contains(0), "contains() must not promote an entry");
    }

    #[test]
    fn reinserting_is_a_touch_not_a_leak() {
        let mut cache = SliceCache::new(usize::MAX, 96);
        cache.insert(1, slice_of(64));
        let once = cache.byte_len();
        cache.insert(1, slice_of(64));
        assert_eq!(cache.byte_len(), once);
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn an_oversized_slice_is_kept_alone() {
        let mut cache = SliceCache::new(1024, 96);
        cache.insert(1, slice_of(512));
        assert_eq!(cache.len(), 1);
        cache.insert(2, slice_of(512));
        assert_eq!(cache.len(), 1);
        assert!(cache.contains(2));
    }
}
