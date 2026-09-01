//! `.xr3d` — a small binary mesh-bundle format for the 3D render view.
//!
//! Segmentation results arrive as a handful of named triangle meshes (one per
//! anatomical structure) plus enough per-group metadata — a colour, a default
//! visibility, an optional volume — to drive the render UI without a second
//! round trip. Rather than reach for a general mesh format (glTF, OBJ, ...)
//! and its dependency weight, `.xr3d` is a flat, little-endian layout that
//! this crate can both read and, for tests, write — see the parser below.
//!
//! # Robustness
//!
//! Like [`crate::parse_header`], this parses bytes a user dropped on the
//! page: every read is bounds-checked against what remains, and a malformed
//! or truncated file returns [`Xr3dError`] rather than panicking. There is no
//! `unsafe` here and there does not need to be.

use std::convert::TryInto;

/// The four magic bytes every `.xr3d` file starts with.
pub const XR3D_MAGIC: &[u8; 4] = b"XR3D";

/// The only format version this build understands.
const XR3D_VERSION: u32 = 1;

/// Upper bound on `group_count`, so a corrupt header cannot make the parser
/// try to allocate an absurd number of empty groups before hitting the first
/// real bounds check.
const MAX_GROUPS: u32 = 4096;

/// One named surface: a triangle mesh plus the metadata the render UI needs.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshGroup {
    pub key: String,
    /// sRGB R, G, B, A — A is `opacity * 255`.
    pub color: [u8; 4],
    /// Bit 0 of the on-disk `flags` word.
    pub visible: bool,
    /// `None` when the stored value is NaN, meaning "unknown".
    pub volume_ml: Option<f32>,
    /// `positions.len() == 3 * vert_count`, LPS millimetres.
    pub positions: Vec<f32>,
    /// `normals.len() == 3 * vert_count`.
    pub normals: Vec<f32>,
    /// `indices.len()` is a multiple of 3; every value is `< vert_count`.
    pub indices: Vec<u32>,
}

/// A parsed `.xr3d` file: every group plus a bounding box over all of them.
#[derive(Debug, Clone, PartialEq)]
pub struct MeshBundle {
    pub groups: Vec<MeshGroup>,
    /// (min, max) over every group's `positions`. `([0.;3], [0.;3])` when the
    /// bundle holds zero vertices in total.
    pub bbox: ([f32; 3], [f32; 3]),
}

/// Everything that can go wrong reading a `.xr3d` file — always a rejection
/// of the file, never a panic.
#[derive(Debug)]
pub enum Xr3dError {
    /// Missing or wrong `XR3D` magic.
    BadMagic,
    /// `version` field was not [`XR3D_VERSION`].
    BadVersion(u32),
    /// The buffer ran out before a field or section it declared could be
    /// read in full.
    Truncated(&'static str),
    /// `group_count` exceeded [`MAX_GROUPS`].
    TooManyGroups(u32),
    /// A group's `key_len` was larger than the bytes remaining in the file —
    /// clearly corrupt, and reading it would try to allocate nonsense.
    KeyTooLarge(u32),
    /// A key's bytes were not valid UTF-8.
    BadKey(std::str::Utf8Error),
    /// `index_count` was not a multiple of 3.
    BadIndexCount(u32),
    /// An index referenced a vertex past the end of that group's positions.
    IndexOutOfRange { index: u32, vert_count: u32 },
}

impl std::fmt::Display for Xr3dError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Xr3dError::BadMagic => write!(f, "not an XR3D file: bad magic"),
            Xr3dError::BadVersion(v) => write!(f, "unsupported XR3D version: {v}"),
            Xr3dError::Truncated(what) => write!(f, "truncated XR3D file: {what}"),
            Xr3dError::TooManyGroups(n) => write!(f, "too many groups: {n} > {MAX_GROUPS}"),
            Xr3dError::KeyTooLarge(n) => write!(f, "group key length {n} exceeds remaining bytes"),
            Xr3dError::BadKey(e) => write!(f, "group key is not valid UTF-8: {e}"),
            Xr3dError::BadIndexCount(n) => {
                write!(f, "index_count {n} is not a multiple of 3")
            }
            Xr3dError::IndexOutOfRange { index, vert_count } => write!(
                f,
                "index {index} is out of range for {vert_count} vertices"
            ),
        }
    }
}

impl std::error::Error for Xr3dError {}

/// True if `prefix` starts with the `.xr3d` magic. `prefix` may be shorter
/// than the magic itself, in which case this is `false` rather than an
/// error — callers use this for a quick sniff, not for validation.
pub fn is_xr3d(prefix: &[u8]) -> bool {
    prefix.len() >= XR3D_MAGIC.len() && &prefix[..XR3D_MAGIC.len()] == XR3D_MAGIC
}

/// A bounds-checked cursor over the file bytes.
///
/// Every accessor either returns the requested value or a [`Xr3dError`] —
/// there is no path from a short read to a panic.
struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { bytes, pos: 0 }
    }

    fn remaining(&self) -> usize {
        self.bytes.len().saturating_sub(self.pos)
    }

    fn take(&mut self, n: usize, what: &'static str) -> Result<&'a [u8], Xr3dError> {
        if self.remaining() < n {
            return Err(Xr3dError::Truncated(what));
        }
        let slice = &self.bytes[self.pos..self.pos + n];
        self.pos += n;
        Ok(slice)
    }

    fn u32(&mut self, what: &'static str) -> Result<u32, Xr3dError> {
        let b = self.take(4, what)?;
        Ok(u32::from_le_bytes(b.try_into().unwrap()))
    }

    fn f32(&mut self, what: &'static str) -> Result<f32, Xr3dError> {
        let b = self.take(4, what)?;
        Ok(f32::from_le_bytes(b.try_into().unwrap()))
    }

    /// Read `count` little-endian `f32`s.
    fn f32_vec(&mut self, count: usize, what: &'static str) -> Result<Vec<f32>, Xr3dError> {
        let bytes = self.take(count.saturating_mul(4), what)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }

    /// Read `count` little-endian `u32`s.
    fn u32_vec(&mut self, count: usize, what: &'static str) -> Result<Vec<u32>, Xr3dError> {
        let bytes = self.take(count.saturating_mul(4), what)?;
        Ok(bytes
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes(c.try_into().unwrap()))
            .collect())
    }
}

/// Number of padding bytes to reach the next 4-byte boundary from `len`.
fn pad_to_4(len: usize) -> usize {
    (4 - (len % 4)) % 4
}

/// Parse a whole `.xr3d` file already in memory.
///
/// Every section is bounds-checked as it is read, so a truncated or
/// adversarial buffer returns an error rather than panicking or reading past
/// the end of `bytes`.
pub fn parse_xr3d(bytes: &[u8]) -> Result<MeshBundle, Xr3dError> {
    let mut r = Reader::new(bytes);

    let magic = r.take(4, "magic")?;
    if magic != XR3D_MAGIC.as_slice() {
        return Err(Xr3dError::BadMagic);
    }
    let version = r.u32("version")?;
    if version != XR3D_VERSION {
        return Err(Xr3dError::BadVersion(version));
    }
    let group_count = r.u32("group_count")?;
    if group_count > MAX_GROUPS {
        return Err(Xr3dError::TooManyGroups(group_count));
    }
    let _reserved = r.u32("reserved")?;

    let mut groups = Vec::with_capacity(group_count as usize);
    let mut min = [f32::INFINITY; 3];
    let mut max = [f32::NEG_INFINITY; 3];
    let mut any_vertex = false;

    for _ in 0..group_count {
        let key_len = r.u32("key_len")?;
        if key_len as usize > r.remaining() {
            return Err(Xr3dError::KeyTooLarge(key_len));
        }
        let key_bytes = r.take(key_len as usize, "key")?;
        let key = std::str::from_utf8(key_bytes)
            .map_err(Xr3dError::BadKey)?
            .to_owned();
        r.take(pad_to_4(key_len as usize), "key padding")?;

        let color_bytes = r.take(4, "color")?;
        let color: [u8; 4] = color_bytes.try_into().unwrap();

        let flags = r.u32("flags")?;
        let visible = flags & 1 != 0;

        let volume_raw = r.f32("volume_ml")?;
        let volume_ml = if volume_raw.is_nan() {
            None
        } else {
            Some(volume_raw)
        };

        let vert_count = r.u32("vert_count")?;
        let index_count = r.u32("index_count")?;
        if index_count % 3 != 0 {
            return Err(Xr3dError::BadIndexCount(index_count));
        }

        let positions = r.f32_vec(vert_count as usize * 3, "positions")?;
        let normals = r.f32_vec(vert_count as usize * 3, "normals")?;
        let indices = r.u32_vec(index_count as usize, "indices")?;

        for &i in &indices {
            if i >= vert_count {
                return Err(Xr3dError::IndexOutOfRange {
                    index: i,
                    vert_count,
                });
            }
        }

        for chunk in positions.chunks_exact(3) {
            any_vertex = true;
            for axis in 0..3 {
                if chunk[axis] < min[axis] {
                    min[axis] = chunk[axis];
                }
                if chunk[axis] > max[axis] {
                    max[axis] = chunk[axis];
                }
            }
        }

        groups.push(MeshGroup {
            key,
            color,
            visible,
            volume_ml,
            positions,
            normals,
            indices,
        });
    }

    let bbox = if any_vertex {
        (min, max)
    } else {
        ([0.0; 3], [0.0; 3])
    };

    Ok(MeshBundle { groups, bbox })
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A minimal in-memory writer for the format, used only by tests. This
    /// also serves as executable documentation of the writer side of the
    /// spec — anyone regenerating `.xr3d` files elsewhere can read it next
    /// to [`parse_xr3d`].
    struct GroupSpec {
        key: &'static str,
        color: [u8; 4],
        flags: u32,
        volume_ml: f32,
        positions: Vec<f32>,
        normals: Vec<f32>,
        indices: Vec<u32>,
    }

    fn write_xr3d(groups: &[GroupSpec]) -> Vec<u8> {
        let mut out = Vec::new();
        out.extend_from_slice(XR3D_MAGIC);
        out.extend_from_slice(&1u32.to_le_bytes()); // version
        out.extend_from_slice(&(groups.len() as u32).to_le_bytes()); // group_count
        out.extend_from_slice(&0u32.to_le_bytes()); // reserved

        for g in groups {
            let key_bytes = g.key.as_bytes();
            out.extend_from_slice(&(key_bytes.len() as u32).to_le_bytes());
            out.extend_from_slice(key_bytes);
            for _ in 0..pad_to_4(key_bytes.len()) {
                out.push(0);
            }

            out.extend_from_slice(&g.color);
            out.extend_from_slice(&g.flags.to_le_bytes());
            out.extend_from_slice(&g.volume_ml.to_le_bytes());

            let vert_count = (g.positions.len() / 3) as u32;
            let index_count = g.indices.len() as u32;
            out.extend_from_slice(&vert_count.to_le_bytes());
            out.extend_from_slice(&index_count.to_le_bytes());

            for v in &g.positions {
                out.extend_from_slice(&v.to_le_bytes());
            }
            for v in &g.normals {
                out.extend_from_slice(&v.to_le_bytes());
            }
            for i in &g.indices {
                out.extend_from_slice(&i.to_le_bytes());
            }
        }

        out
    }

    /// Two quads' worth of geometry (4 verts / 2 tris) for the first group,
    /// one triangle (3 verts / 1 tri) for the second — enough to exercise
    /// multiple groups, non-trivial index buffers, and a key whose length
    /// (11 bytes -> "kidney_left") is not a multiple of 4.
    fn two_group_bundle() -> Vec<GroupSpec> {
        vec![
            GroupSpec {
                key: "kidney_left",
                color: [200, 50, 50, 180],
                flags: 1, // visible
                volume_ml: 123.5,
                positions: vec![
                    0.0, 0.0, 0.0, //
                    1.0, 0.0, 0.0, //
                    1.0, 1.0, 0.0, //
                    0.0, 1.0, 0.0,
                ],
                normals: vec![
                    0.0, 0.0, 1.0, //
                    0.0, 0.0, 1.0, //
                    0.0, 0.0, 1.0, //
                    0.0, 0.0, 1.0,
                ],
                indices: vec![0, 1, 2, 0, 2, 3],
            },
            GroupSpec {
                key: "aorta",
                color: [220, 20, 20, 255],
                flags: 0, // hidden by default
                volume_ml: f32::NAN,
                positions: vec![
                    -5.0, -5.0, -5.0, //
                    5.0, -5.0, -5.0, //
                    0.0, 5.0, -5.0,
                ],
                normals: vec![
                    0.0, 0.0, -1.0, //
                    0.0, 0.0, -1.0, //
                    0.0, 0.0, -1.0,
                ],
                indices: vec![0, 1, 2],
            },
        ]
    }

    #[test]
    fn round_trips_a_two_group_bundle() {
        let spec = two_group_bundle();
        let bytes = write_xr3d(&spec);
        let bundle = parse_xr3d(&bytes).expect("valid bundle should parse");

        assert_eq!(bundle.groups.len(), 2);

        let g0 = &bundle.groups[0];
        assert_eq!(g0.key, "kidney_left");
        assert_eq!(g0.color, [200, 50, 50, 180]);
        assert!(g0.visible);
        assert_eq!(g0.volume_ml, Some(123.5));
        assert_eq!(g0.positions, spec[0].positions);
        assert_eq!(g0.normals, spec[0].normals);
        assert_eq!(g0.indices, vec![0, 1, 2, 0, 2, 3]);

        let g1 = &bundle.groups[1];
        assert_eq!(g1.key, "aorta");
        assert!(!g1.visible, "flags bit 0 clear means hidden by default");
        assert_eq!(g1.volume_ml, None, "NaN on disk must decode to None");
        assert_eq!(g1.indices, vec![0, 1, 2]);

        // bbox over both groups: min/max across all 7 vertices.
        assert_eq!(bundle.bbox.0, [-5.0, -5.0, -5.0]);
        assert_eq!(bundle.bbox.1, [5.0, 5.0, 0.0]);
    }

    #[test]
    fn is_xr3d_checks_the_magic_only() {
        assert!(is_xr3d(b"XR3D"));
        assert!(is_xr3d(b"XR3Dxxxxxxxx"));
        assert!(!is_xr3d(b"XR3"), "too short to contain the magic");
        assert!(!is_xr3d(b""));
        assert!(!is_xr3d(b"DICM"), "wrong magic entirely");
    }

    #[test]
    fn rejects_bad_magic() {
        let mut bytes = write_xr3d(&two_group_bundle());
        bytes[0] = b'X';
        bytes[1] = b'X';
        match parse_xr3d(&bytes) {
            Err(Xr3dError::BadMagic) => {}
            other => panic!("expected BadMagic, got {other:?}"),
        }
    }

    #[test]
    fn rejects_wrong_version() {
        let mut bytes = write_xr3d(&two_group_bundle());
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        match parse_xr3d(&bytes) {
            Err(Xr3dError::BadVersion(2)) => {}
            other => panic!("expected BadVersion(2), got {other:?}"),
        }
    }

    #[test]
    fn rejects_index_out_of_range() {
        let mut spec = two_group_bundle();
        spec[0].indices[0] = 99; // vert_count is 4
        let bytes = write_xr3d(&spec);
        match parse_xr3d(&bytes) {
            Err(Xr3dError::IndexOutOfRange { index: 99, .. }) => {}
            other => panic!("expected IndexOutOfRange, got {other:?}"),
        }
    }

    #[test]
    fn rejects_index_count_not_multiple_of_three() {
        // Hand-build a single-group file with a bad index_count, bypassing
        // the writer helper (which always emits full triangles).
        let mut out = Vec::new();
        out.extend_from_slice(XR3D_MAGIC);
        out.extend_from_slice(&1u32.to_le_bytes());
        out.extend_from_slice(&1u32.to_le_bytes()); // group_count
        out.extend_from_slice(&0u32.to_le_bytes());

        let key = b"x";
        out.extend_from_slice(&(key.len() as u32).to_le_bytes());
        out.extend_from_slice(key);
        out.extend_from_slice(&[0, 0, 0]); // pad "x" (1 byte) to 4

        out.extend_from_slice(&[0, 0, 0, 255]); // color
        out.extend_from_slice(&1u32.to_le_bytes()); // flags
        out.extend_from_slice(&0f32.to_le_bytes()); // volume_ml
        out.extend_from_slice(&3u32.to_le_bytes()); // vert_count
        out.extend_from_slice(&4u32.to_le_bytes()); // index_count: not a multiple of 3

        match parse_xr3d(&out) {
            Err(Xr3dError::BadIndexCount(4)) => {}
            other => panic!("expected BadIndexCount(4), got {other:?}"),
        }
    }

    #[test]
    fn truncated_buffer_errors_instead_of_panicking() {
        let bytes = write_xr3d(&two_group_bundle());
        // Try every prefix length; each must either parse (only at the full
        // length) or return a clean error — never panic.
        for len in 0..bytes.len() {
            let prefix = &bytes[..len];
            let result = std::panic::catch_unwind(|| parse_xr3d(prefix));
            let result = result.expect("parse_xr3d must not panic on truncated input");
            assert!(
                result.is_err(),
                "prefix of length {len} unexpectedly parsed successfully"
            );
        }
        // And the full buffer must succeed.
        assert!(parse_xr3d(&bytes).is_ok());
    }

    #[test]
    fn empty_buffer_errors() {
        assert!(parse_xr3d(&[]).is_err());
    }

    #[test]
    fn zero_groups_gives_empty_bundle_with_zero_bbox() {
        let bytes = write_xr3d(&[]);
        let bundle = parse_xr3d(&bytes).expect("zero-group bundle is valid");
        assert!(bundle.groups.is_empty());
        assert_eq!(bundle.bbox, ([0.0; 3], [0.0; 3]));
    }
}
