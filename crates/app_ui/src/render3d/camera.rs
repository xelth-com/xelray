//! Orbit camera math for the wgpu 3D viewer.
//!
//! ## Coordinate conventions
//!
//! Mesh vertices arrive in **LPS millimetres** — DICOM's patient-anatomical
//! frame: +x = patient Left, +y = Posterior, +z = Superior (toward the
//! head). This is a right-handed frame, but it is not the frame we display
//! in: the old plotly-based viewer plotted `(x, -y, z)`, and this camera
//! preserves that convention so the two viewers agree pixel-for-pixel.
//!
//! We call `(x, -y, z)` **display space**. It is also right-handed, with
//! +z still up. All camera state below (`target`, [`OrbitCamera::eye`],
//! ...) is expressed in display space.
//!
//! The LPS -> display transform is the fixed flip `S = diag(1, -1, 1)`.
//! Rather than transform every vertex on the CPU, [`OrbitCamera::view`]
//! folds `S` into the view matrix: `view() = look_at(eye, target, +z) * S`.
//! Feed it LPS-space vertices directly and the flip happens for free in the
//! vertex shader. **Normals need the same flip** for lighting to stay
//! correct — the renderer is responsible for applying `S` (equivalently its
//! transpose-inverse, which for a pure axis flip is `S` itself) to normals
//! before lighting. Because `S` has a negative determinant it also flips
//! triangle winding; the renderer must run with back-face culling disabled,
//! so winding is irrelevant.
//!
//! ## wgpu / WebGPU NDC
//!
//! [`OrbitCamera::proj`] uses [`Mat4::perspective_rh`] — right-handed view
//! space, depth range `0..1` — matching wgpu/WebGPU, *not* the OpenGL
//! `-1..1` convention ([`Mat4::perspective_rh_gl`]).

use glam::{Mat4, Vec3};

/// Elevation is clamped to keep the camera from flipping over the pole.
const MAX_ELEVATION: f32 = 89.0 / 180.0 * std::f32::consts::PI;

/// Orbit sensitivity: radians of azimuth/elevation per pixel of drag.
const ORBIT_SENSITIVITY: f32 = 0.008;

/// An orbiting camera: an eye that circles `target` at fixed `distance`,
/// parameterized by `azimuth` (rotation around +z) and `elevation`
/// (angle above/below the xy-plane). Everything here lives in **display
/// space** — see the module docs.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct OrbitCamera {
    /// Look-at point, in display space.
    pub target: Vec3,
    /// Radians, measured around +z. Wraps to (-pi, pi].
    pub azimuth: f32,
    /// Radians above the xy-plane, clamped to +-89 degrees.
    pub elevation: f32,
    /// Distance from `target` to the eye.
    pub distance: f32,
    /// Vertical field of view, radians. Defaults to ~45 degrees.
    pub fov_y: f32,
    pub near: f32,
    pub far: f32,

    /// The distance [`OrbitCamera::fit`] chose, which bounds
    /// [`OrbitCamera::zoom`].
    ///
    /// Recentring re-runs `fit` over whatever is visible rather than restoring
    /// a stored pose, so nothing else about the initial framing is kept: the
    /// box worth framing changes every time an organ is toggled.
    fit_distance: f32,
}

impl OrbitCamera {
    /// Frame a bounding box (given in **LPS millimetres**) with a default
    /// posterior-oblique orientation, matching the old plotly viewer's
    /// initial pose.
    ///
    /// `target` becomes the box's center, converted to display space (the
    /// y-flip is applied here). `distance` is picked so the bounding
    /// sphere fits within `fov_y` with a 20% margin; `near`/`far` are
    /// derived from the sphere radius.
    pub fn fit(bbox_min: [f32; 3], bbox_max: [f32; 3]) -> Self {
        let min = Vec3::from(bbox_min);
        let max = Vec3::from(bbox_max);

        let center_lps = (min + max) * 0.5;
        let target = Vec3::new(center_lps.x, -center_lps.y, center_lps.z);

        let radius = ((max - min).length() * 0.5).max(1e-3);

        let fov_y = 45.0_f32.to_radians();
        let distance = radius / (fov_y * 0.5).tan() * 1.2;
        let near = (distance / 100.0).max(0.1);
        let far = distance + 4.0 * radius;

        // Old viewer's default eye direction in display space:
        // posterior-oblique from the patient's left, +z up.
        let dir = Vec3::new(1.35, -1.55, 0.5).normalize();
        let azimuth = dir.y.atan2(dir.x);
        let elevation = dir.z.clamp(-1.0, 1.0).asin().clamp(-MAX_ELEVATION, MAX_ELEVATION);

        Self {
            target,
            azimuth,
            elevation,
            distance,
            fov_y,
            near,
            far,
            fit_distance: distance,
        }
    }

    /// Rotate around `target`: `dx_px`/`dy_px` are mouse-drag deltas in
    /// pixels.
    pub fn orbit(&mut self, dx_px: f32, dy_px: f32) {
        self.azimuth = wrap_angle(self.azimuth - dx_px * ORBIT_SENSITIVITY);
        self.elevation = (self.elevation + dy_px * ORBIT_SENSITIVITY)
            .clamp(-MAX_ELEVATION, MAX_ELEVATION);
    }

    /// Translate `target` within the camera's view plane, in response to a
    /// drag of `dx_px`/`dy_px` pixels over a viewport `viewport_h_px`
    /// pixels tall.
    pub fn pan(&mut self, dx_px: f32, dy_px: f32, viewport_h_px: f32) {
        let world_per_px =
            2.0 * self.distance * (self.fov_y * 0.5).tan() / viewport_h_px.max(1.0);
        let (_, right, up) = self.view_basis();
        self.target -= right * dx_px * world_per_px;
        self.target += up * dy_px * world_per_px;
    }

    /// Scale `distance` by `factor` (>1 zooms out, <1 zooms in), clamped to
    /// [0.05x, 20x] of the distance computed by [`OrbitCamera::fit`].
    pub fn zoom(&mut self, factor: f32) {
        let min = self.fit_distance * 0.05;
        let max = self.fit_distance * 20.0;
        self.distance = (self.distance * factor).clamp(min, max);
    }

    /// Eye position in display space.
    pub fn eye(&self) -> Vec3 {
        self.target + self.offset()
    }

    /// `look_at(eye, target, +z) * S`, where `S = diag(1, -1, 1)` is the
    /// LPS -> display flip. Apply this directly to LPS-space vertex
    /// positions — see the module docs.
    pub fn view(&self) -> Mat4 {
        let flip = Mat4::from_scale(Vec3::new(1.0, -1.0, 1.0));
        Mat4::look_at_rh(self.eye(), self.target, Vec3::Z) * flip
    }

    /// wgpu/WebGPU perspective projection (right-handed, depth `0..1`).
    pub fn proj(&self, aspect: f32) -> Mat4 {
        Mat4::perspective_rh(self.fov_y, aspect, self.near, self.far)
    }

    /// `proj(aspect) * view()`.
    pub fn view_proj(&self, aspect: f32) -> Mat4 {
        self.proj(aspect) * self.view()
    }

    /// Vector from `target` to `eye`.
    fn offset(&self) -> Vec3 {
        Vec3::new(
            self.elevation.cos() * self.azimuth.cos(),
            self.elevation.cos() * self.azimuth.sin(),
            self.elevation.sin(),
        ) * self.distance
    }

    /// (forward, right, up) basis for the current view, in display space.
    fn view_basis(&self) -> (Vec3, Vec3, Vec3) {
        let forward = -self.offset().normalize();
        let right = forward.cross(Vec3::Z).normalize_or_zero();
        let up = right.cross(forward);
        (forward, right, up)
    }
}

/// Wrap an angle in radians to `(-pi, pi]`.
fn wrap_angle(a: f32) -> f32 {
    use std::f32::consts::PI;
    let two_pi = 2.0 * PI;
    let wrapped = a.rem_euclid(two_pi);
    if wrapped > PI {
        wrapped - two_pi
    } else {
        wrapped
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fit_centers_bbox_and_view_proj_centers_target() {
        let cam = OrbitCamera::fit([-10.0, -10.0, -10.0], [10.0, 10.0, 10.0]);
        assert_eq!(cam.target, Vec3::ZERO);

        // target is in display space; convert back to LPS before feeding
        // it through the camera (which expects LPS-space vertices).
        let target_lps = Vec3::new(cam.target.x, -cam.target.y, cam.target.z);
        let clip = cam.view_proj(1.0) * target_lps.extend(1.0);

        assert!(clip.w > 0.0, "w = {}", clip.w);
        let ndc_x = clip.x / clip.w;
        let ndc_y = clip.y / clip.w;
        assert!(ndc_x.abs() < 1e-4, "ndc_x = {ndc_x}");
        assert!(ndc_y.abs() < 1e-4, "ndc_y = {ndc_y}");
    }

    #[test]
    fn default_eye_direction_matches_old_viewer() {
        let cam = OrbitCamera::fit([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let dir = (cam.eye() - cam.target).normalize();
        let expected = Vec3::new(1.35, -1.55, 0.5).normalize();
        assert!((dir - expected).length() < 1e-4, "dir = {dir:?}");
    }

    #[test]
    fn elevation_clamps() {
        let mut cam = OrbitCamera::fit([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        cam.orbit(0.0, 1_000_000.0);
        assert!(cam.elevation <= MAX_ELEVATION + 1e-5);
        cam.orbit(0.0, -2_000_000.0);
        assert!(cam.elevation >= -MAX_ELEVATION - 1e-5);
    }

    #[test]
    fn zoom_clamps() {
        let mut cam = OrbitCamera::fit([0.0, 0.0, 0.0], [1.0, 1.0, 1.0]);
        let fit_distance = cam.fit_distance;

        cam.zoom(1e6);
        assert!((cam.distance - fit_distance * 20.0).abs() < 1e-3);

        cam.zoom(1e-9);
        assert!((cam.distance - fit_distance * 0.05).abs() < 1e-3);
    }

    #[test]
    fn y_flip_applied_in_view() {
        let cam = OrbitCamera::fit([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);

        // No-flip lookAt, applied to a point already in display space.
        let look = Mat4::look_at_rh(cam.eye(), cam.target, Vec3::Z);
        let expected_display = Vec3::new(0.0, -10.0, 0.0);
        let via_look = look * expected_display.extend(1.0);

        // Camera's view(), which folds S in, applied to the equivalent
        // LPS-space point.
        let lps_point = Vec3::new(0.0, 10.0, 0.0);
        let via_view = cam.view() * lps_point.extend(1.0);

        assert!((via_view - via_look).length() < 1e-4);
    }

    #[test]
    fn target_depth_is_in_unit_range() {
        let cam = OrbitCamera::fit([-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]);
        let target_lps = Vec3::new(cam.target.x, -cam.target.y, cam.target.z);
        let clip = cam.view_proj(1.33) * target_lps.extend(1.0);
        let ndc_z = clip.z / clip.w;
        assert!(ndc_z > 0.0 && ndc_z < 1.0, "ndc_z = {ndc_z}");
    }
}
