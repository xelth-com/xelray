# xelray/ml — local MONAI CT segmentation test bench

A throwaway-but-reproducible bench for running the MONAI model-zoo bundle
[`wholeBody_ct_segmentation`](https://huggingface.co/MONAI/wholeBody_ct_segmentation)
against a real DICOM CT study **entirely on this machine, CPU only**.

The point is not the segmentation itself — it is the **benchmark**: how long does a
104-organ 3D CNN actually take on ordinary consumer CPU hardware, and how much RAM
does it need? That number decides whether xelray can plausibly run this kind of model
in the browser (ONNX Runtime Web / WASM / WebGPU) or whether it has to stay server-side.

No patient data, model weights, or output images are committed. See `.gitignore`.

## Model

`wholeBody_ct_segmentation` is a SegResNet (18.8 M params, 3D, 1 input channel,
**105 output channels** = background + 104 TotalSegmentator classes). The bundle ships
two checkpoints:

| checkpoint        | spacing        | size   |
|-------------------|----------------|--------|
| `model.pt`        | 1.5 mm iso     | ~230 MB |
| `model_lowres.pt` | 3.0 mm iso     | ~75 MB  |

This bench defaults to the **low-res 3 mm** model (`--pixdim 3.0`); pass `--pixdim 1.5`
to use the high-res one instead.

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cpu
.\.venv\Scripts\python.exe -m pip install "monai[nibabel]" SimpleITK matplotlib tqdm fire requests
.\.venv\Scripts\python.exe -m monai.bundle download --name wholeBody_ct_segmentation --bundle_dir bundle
```

`fire` and `requests` are only needed for `monai.bundle download`, not for inference.

## Run

```powershell
.\.venv\Scripts\python.exe segment.py `
    --dicom "<path to a DICOM series folder>" `
    --bundle bundle\wholeBody_ct_segmentation `
    --out out --threads 4 --overlap 0.5
```

Three separately cached stages, so a long run can be resumed with `--stage`:

| stage   | does                                                        | writes |
|---------|-------------------------------------------------------------|--------|
| `prep`  | SimpleITK reads the DICOM series → NIfTI, then the bundle's exact preprocessing chain (RAS → 3 mm resample → `NormalizeIntensity(nonzero)` → `ScaleIntensity(-1, 1)`) | `ct_orig.nii.gz`, `prep.npz` |
| `infer` | `SlidingWindowInferer`, 96³ ROI, gaussian blending, CPU      | `pred_model_space.nii.gz` |
| `post`  | nearest-neighbour resample back to the original CT grid, per-organ volumes, overlay PNGs | `pred_orig_space.nii.gz`, `volumes.json`, `overlay_*.png`, `zoom_*.png` |

Wall-clock for each stage lands in `out/timings.json`.

## Notes / deviations from the bundle reference pipeline

* The bundle's `configs/inference.json` drives everything through
  `SupervisedEvaluator` + `Invertd` + `SaveImaged` over a folder of `.nii.gz`.
  `segment.py` reimplements the same transform chain directly so it can take a DICOM
  directory as input and so each stage can be cached — the **transforms, ROI size,
  blending mode, padding mode and network definition are copied verbatim** from that
  config.
* Instead of MONAI's `Invertd`, the label map is resampled back onto the original CT
  grid with SimpleITK nearest-neighbour. Same result, and it keeps the full-resolution
  label volume as a real NIfTI in the scanner's own geometry.
* `--overlap` defaults to `0.5`; the bundle's own default is `0.25`. Higher overlap =
  more sliding windows = slower but smoother seams.
* `amp=True` in the bundle is a CUDA setting and is not used here.
* Label indices come from `bundle/wholeBody_ct_segmentation/configs/metadata.json`
  → `network_data_format.outputs.pred.channel_def`. The subset reported on is in
  `ORGANS` at the top of `segment.py`.

## Orientation convention in the overlays

Arrays are `(z, y, x)` in LPS. Overlays are rendered with matplotlib's default
`origin="upper"`, which gives the radiological view: **anterior up, patient's LEFT on
the image right**. Increasing `z` index is **superior**.
