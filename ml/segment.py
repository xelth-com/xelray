"""Local CPU test bench for MONAI `wholeBody_ct_segmentation` on a real DICOM CT study.

Runs the MONAI model-zoo bundle's reference inference pipeline entirely offline on
CPU, against a DICOM series read with SimpleITK, and produces verification
artifacts (per-organ volumes, overlay PNGs, a markdown report).

Stages are separately cached under --out so a long CPU run can be resumed:

    stage 1  prep    DICOM series -> NIfTI -> bundle preprocessing  (ct_orig.nii.gz, prep.npz)
    stage 2  infer   sliding-window SegResNet -> argmax label map   (pred_model_space.nii.gz)
    stage 3  post    resample to original grid, volumes, overlays   (volumes.json, *.png, report.md)

Usage:
    python segment.py --dicom <series_dir> --bundle <bundle_dir> --out out
    python segment.py ... --stage post          # re-run only postprocessing
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

# Keep BLAS/OMP from oversubscribing the 4 physical cores before torch is imported.
_THREADS = os.environ.get("XELRAY_THREADS")
if _THREADS:
    for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(_v, _THREADS)

import SimpleITK as sitk  # noqa: E402
import torch  # noqa: E402
from monai.inferers import SlidingWindowInferer  # noqa: E402
from monai.networks.nets import SegResNet  # noqa: E402
from monai.transforms import (  # noqa: E402
    Compose,
    EnsureChannelFirstd,
    EnsureTyped,
    LoadImaged,
    NormalizeIntensityd,
    Orientationd,
    ScaleIntensityd,
    Spacingd,
)

# Organs we report on. Indices come from configs/metadata.json -> channel_def.
ORGANS = {
    "spleen": 1,
    "kidney_right": 2,
    "kidney_left": 3,
    "gallbladder": 4,
    "liver": 5,
    "aorta": 7,
    "pancreas": 10,
    "adrenal_gland_right": 11,
    "adrenal_gland_left": 12,
    "urinary_bladder": 104,
}

# RGB overlay colors, one per reported organ.
COLORS = {
    "spleen": (0.20, 0.80, 0.35),
    "kidney_right": (0.20, 0.45, 1.00),
    "kidney_left": (1.00, 0.30, 0.25),
    "gallbladder": (0.95, 0.85, 0.20),
    "liver": (0.85, 0.45, 0.90),
    "aorta": (1.00, 0.55, 0.10),
    "pancreas": (0.20, 0.90, 0.90),
    "adrenal_gland_right": (0.55, 0.35, 0.10),
    "adrenal_gland_left": (0.75, 0.55, 0.20),
    "urinary_bladder": (0.60, 0.60, 0.60),
}


def log(msg: str) -> None:
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def peak_rss_mb() -> float | None:
    """Peak working set of this process, in MB. Windows only; None elsewhere."""
    if sys.platform != "win32":
        return None
    import ctypes
    from ctypes import wintypes

    class PMC(ctypes.Structure):
        _fields_ = [
            ("cb", wintypes.DWORD),
            ("PageFaultCount", wintypes.DWORD),
            ("PeakWorkingSetSize", ctypes.c_size_t),
            ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
            ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t),
            ("PeakPagefileUsage", ctypes.c_size_t),
        ]

    c = PMC()
    c.cb = ctypes.sizeof(c)
    ctypes.windll.kernel32.GetCurrentProcess.restype = wintypes.HANDLE
    handle = ctypes.windll.kernel32.GetCurrentProcess()
    # Modern Windows exports this from kernel32 as K32...; psapi.dll is the older name.
    for dll, fn in (
        (ctypes.windll.kernel32, "K32GetProcessMemoryInfo"),
        (ctypes.windll.psapi, "GetProcessMemoryInfo"),
    ):
        try:
            f = getattr(dll, fn)
        except AttributeError:
            continue
        f.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
        f.restype = wintypes.BOOL
        if f(handle, ctypes.byref(c), c.cb):
            break
    else:
        return None
    return round(c.PeakWorkingSetSize / (1024 * 1024), 1)


# --------------------------------------------------------------------------- stage 1


def read_dicom_series(dicom_dir: Path, nifti_path: Path) -> sitk.Image:
    """Read an uncompressed DICOM series with SimpleITK and cache it as NIfTI."""
    reader = sitk.ImageSeriesReader()
    series_ids = reader.GetGDCMSeriesIDs(str(dicom_dir))
    if not series_ids:
        raise SystemExit(f"no DICOM series found in {dicom_dir}")
    if len(series_ids) > 1:
        log(f"WARNING: {len(series_ids)} series in {dicom_dir}, using the largest")
    best = max(
        series_ids,
        key=lambda sid: len(reader.GetGDCMSeriesFileNames(str(dicom_dir), sid)),
    )
    files = reader.GetGDCMSeriesFileNames(str(dicom_dir), best)
    reader.SetFileNames(files)
    img = reader.Execute()
    log(
        f"read {len(files)} slices  size={img.GetSize()}  "
        f"spacing={tuple(round(s, 4) for s in img.GetSpacing())}"
    )
    sitk.WriteImage(sitk.Cast(img, sitk.sitkInt16), str(nifti_path), True)
    return img


def stage_prep(args, out: Path) -> dict:
    t0 = time.perf_counter()
    ct_path = out / "ct_orig.nii.gz"
    if ct_path.exists() and not args.force:
        log(f"reusing {ct_path.name}")
    else:
        read_dicom_series(Path(args.dicom), ct_path)
    t_read = time.perf_counter() - t0

    # Exactly the bundle's configs/inference.json `preprocessing` chain,
    # with pixdim taken from the low-res branch (3.0 mm isotropic).
    pre = Compose(
        [
            LoadImaged(keys="image"),
            EnsureTyped(keys="image"),
            EnsureChannelFirstd(keys="image"),
            Orientationd(keys="image", axcodes="RAS"),
            Spacingd(keys="image", pixdim=[args.pixdim] * 3, mode="bilinear"),
            NormalizeIntensityd(keys="image", nonzero=True),
            ScaleIntensityd(keys="image", minv=-1.0, maxv=1.0),
        ]
    )
    t1 = time.perf_counter()
    data = pre({"image": str(ct_path)})
    img = data["image"]
    t_transform = time.perf_counter() - t1

    affine = np.asarray(img.meta["affine"], dtype=np.float64)
    np.savez_compressed(
        out / "prep.npz",
        image=np.asarray(img, dtype=np.float32),
        affine=affine,
    )
    log(f"preprocessed volume {tuple(img.shape)} at {args.pixdim} mm  ({t_transform:.1f}s)")
    return {
        "dicom_read_s": round(t_read, 2),
        "preprocess_s": round(t_transform, 2),
        "model_space_shape": list(map(int, img.shape[1:])),
    }


# --------------------------------------------------------------------------- stage 2


def stage_infer(args, out: Path) -> dict:
    blob = np.load(out / "prep.npz")
    image = torch.from_numpy(blob["image"]).unsqueeze(0)  # 1,1,H,W,D
    affine = blob["affine"]

    threads = args.threads or (os.cpu_count() or 4) // 2 or 1
    torch.set_num_threads(threads)
    log(f"torch threads={threads}  input={tuple(image.shape)}")

    net = SegResNet(
        spatial_dims=3,
        in_channels=1,
        out_channels=105,
        init_filters=32,
        blocks_down=[1, 2, 2, 4],
        blocks_up=[1, 1, 1],
        dropout_prob=0.2,
    )
    ckpt_name = "model.pt" if args.pixdim < 3.0 else "model_lowres.pt"
    ckpt = torch.load(
        Path(args.bundle) / "models" / ckpt_name, map_location="cpu", weights_only=True
    )
    net.load_state_dict(ckpt)
    net.eval()
    log(f"loaded {ckpt_name} ({sum(p.numel() for p in net.parameters())/1e6:.1f}M params)")

    inferer = SlidingWindowInferer(
        roi_size=[96, 96, 96],
        sw_batch_size=1,
        overlap=args.overlap,
        padding_mode="replicate",
        mode="gaussian",
        device=torch.device("cpu"),
        progress=True,
    )

    t0 = time.perf_counter()
    with torch.no_grad():
        logits = inferer(image, net)
    t_infer = time.perf_counter() - t0
    log(f"sliding-window inference done in {t_infer:.1f}s ({t_infer/60:.1f} min)")

    label = torch.argmax(logits, dim=1).squeeze(0).to(torch.uint8).numpy()
    del logits

    save_label_nifti(label, affine, out / "pred_model_space.nii.gz")
    peak = peak_rss_mb()
    log(f"peak working set: {peak} MB")
    return {
        "inference_s": round(t_infer, 2),
        "overlap": args.overlap,
        "threads": threads,
        "infer_peak_rss_mb": peak,
    }


def save_label_nifti(label: np.ndarray, affine: np.ndarray, path: Path) -> None:
    """Write a label array + RAS affine as NIfTI (nibabel keeps the RAS convention)."""
    import nibabel as nib

    nib.save(nib.Nifti1Image(label.astype(np.uint8), affine), str(path))
    log(f"wrote {path.name} {label.shape}")


# --------------------------------------------------------------------------- stage 3


def stage_post(args, out: Path) -> dict:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.patches as mpatches
    import matplotlib.pyplot as plt

    ct = sitk.ReadImage(str(out / "ct_orig.nii.gz"))
    pred = sitk.ReadImage(str(out / "pred_model_space.nii.gz"))

    t0 = time.perf_counter()
    res = sitk.ResampleImageFilter()
    res.SetReferenceImage(ct)
    res.SetInterpolator(sitk.sitkNearestNeighbor)
    res.SetDefaultPixelValue(0)
    pred_hi = res.Execute(sitk.Cast(pred, sitk.sitkUInt8))
    t_resample = time.perf_counter() - t0
    sitk.WriteImage(pred_hi, str(out / "pred_orig_space.nii.gz"), True)

    ct_np = sitk.GetArrayFromImage(ct)  # z, y, x
    lab_np = sitk.GetArrayFromImage(pred_hi)
    lab_lo = sitk.GetArrayFromImage(pred)

    vox_hi = float(np.prod(ct.GetSpacing()))
    vox_lo = float(np.prod(pred.GetSpacing()))

    volumes = {}
    for name, idx in ORGANS.items():
        n_hi = int((lab_np == idx).sum())
        n_lo = int((lab_lo == idx).sum())
        volumes[name] = {
            "label": idx,
            "voxels_model_space": n_lo,
            "volume_ml_model_space": round(n_lo * vox_lo / 1000.0, 1),
            "voxels_orig_space": n_hi,
            "volume_ml_orig_space": round(n_hi * vox_hi / 1000.0, 1),
        }
    present = sorted(int(v) for v in np.unique(lab_lo) if v != 0)
    payload = {
        "ct_spacing_mm": [round(s, 4) for s in ct.GetSpacing()],
        "ct_size": list(ct.GetSize()),
        "model_spacing_mm": [round(s, 4) for s in pred.GetSpacing()],
        "model_size": list(pred.GetSize()),
        "resample_to_orig_s": round(t_resample, 2),
        "n_labels_predicted": len(present),
        "organs": volumes,
    }
    (out / "volumes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    log(f"wrote volumes.json ({len(present)}/104 labels present)")

    # ---- overlays -------------------------------------------------------
    png_dir = out
    kid = np.isin(lab_np, [ORGANS["kidney_left"], ORGANS["kidney_right"]])
    kid_z = np.where(kid.any(axis=(1, 2)))[0]
    liver_z = np.where((lab_np == ORGANS["liver"]).any(axis=(1, 2)))[0]

    picks: list[tuple[str, int]] = []
    if kid_z.size:
        for i, z in enumerate(np.linspace(kid_z[0], kid_z[-1], 6).round().astype(int)):
            picks.append((f"kidney_{i:02d}_z{z:04d}", int(z)))
    if liver_z.size:
        z = int(liver_z[len(liver_z) // 2])
        picks.append((f"liver_mid_z{z:04d}", z))

    paths = []
    for tag, z in picks:
        p = render_overlay(ct_np, lab_np, z, png_dir / f"overlay_{tag}.png", plt, mpatches)
        paths.append(str(p))

    # Zoomed views of the left renal fossa, sampled below and through the left kidney
    # label -- this is where a lower-pole mass shows up as a defect in the mask.
    lk = lab_np == ORGANS["kidney_left"]
    if lk.any():
        lz, ly, lx = np.nonzero(lk)
        box = (
            max(0, ly.min() - 60),
            min(lab_np.shape[1], ly.max() + 60),
            max(0, lx.min() - 60),
            min(lab_np.shape[2], lx.max() + 60),
        )
        span = lz.max() - lz.min()
        zs = np.unique(
            np.clip(
                np.linspace(lz.min() - 0.45 * span, lz.max(), 6).round().astype(int),
                0,
                lab_np.shape[0] - 1,
            )
        )
        for z in zs:
            p = render_overlay(
                ct_np, lab_np, int(z), png_dir / f"zoom_leftkidney_z{int(z):04d}.png",
                plt, mpatches, crop=box,
            )
            paths.append(str(p))
    log(f"wrote {len(paths)} overlay PNGs")
    payload["overlays"] = paths
    (out / "volumes.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return {"overlays": paths, "volumes": volumes, "payload": payload}


def render_overlay(ct_np, lab_np, z, path, plt, mpatches, wl=50, ww=400, crop=None):
    """Axial CT slice in a soft-tissue window with semi-transparent organ masks.

    `crop` is an optional (y0, y1, x0, x1) index box for a zoomed view.
    """
    if crop is not None:
        y0, y1, x0, x1 = crop
        ct_np = ct_np[:, y0:y1, x0:x1]
        lab_np = lab_np[:, y0:y1, x0:x1]
    sl = ct_np[z].astype(np.float32)
    lo, hi = wl - ww / 2.0, wl + ww / 2.0
    gray = np.clip((sl - lo) / (hi - lo), 0, 1)
    rgb = np.repeat(gray[..., None], 3, axis=2)

    ls = lab_np[z]
    handles = []
    for name, idx in ORGANS.items():
        m = ls == idx
        if not m.any():
            continue
        c = np.array(COLORS[name], dtype=np.float32)
        rgb[m] = 0.55 * rgb[m] + 0.45 * c
        handles.append(mpatches.Patch(color=c, label=f"{name} ({int(m.sum())} px)"))

    fig, ax = plt.subplots(figsize=(8.5, 7.0), dpi=130)
    # Array axes are (z, y, x) in LPS: +y = posterior, +x = patient-left. matplotlib's
    # default origin="upper" therefore gives the radiological view (anterior up, patient
    # left on the image right).
    ax.imshow(rgb)
    ax.set_title(f"axial slice z={z}  (soft tissue W{ww}/L{wl})  patient LEFT = image right")
    ax.axis("off")
    if handles:
        ax.legend(handles=handles, loc="upper left", fontsize=7, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    return path


# --------------------------------------------------------------------------- main


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dicom", required=True, help="DICOM series directory")
    ap.add_argument("--bundle", required=True, help="wholeBody_ct_segmentation bundle dir")
    ap.add_argument("--out", default="out")
    ap.add_argument("--pixdim", type=float, default=3.0, help="3.0 = lowres model, 1.5 = highres")
    ap.add_argument("--overlap", type=float, default=0.5)
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--stage", default="all", choices=["all", "prep", "infer", "post"])
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    timings_path = out / "timings.json"
    timings = json.loads(timings_path.read_text()) if timings_path.exists() else {}

    stages = ["prep", "infer", "post"] if args.stage == "all" else [args.stage]
    for s in stages:
        log(f"=== stage: {s} ===")
        if s == "prep":
            timings.update(stage_prep(args, out))
        elif s == "infer":
            timings.update(stage_infer(args, out))
        else:
            r = stage_post(args, out)
            timings["resample_to_orig_s"] = r["payload"]["resample_to_orig_s"]
        timings_path.write_text(json.dumps(timings, indent=2), encoding="utf-8")
    log(f"timings: {json.dumps(timings)}")


if __name__ == "__main__":
    sys.exit(main())
