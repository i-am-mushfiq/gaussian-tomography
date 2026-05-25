# gaussian-tomography
## Internal Handoff — R²-Gaussian for CT Brain Reconstruction

**Date**: May 2026  
**Author of handoff**: Derived from full repository audit  
**Intended audience**: Incoming research/engineering team taking over the project  

---

## GitHub Repositories

| Repo | Purpose | Fork | Upstream |
|------|---------|------|----------|
| **r2_gaussian** | Primary research codebase | [i-am-mushfiq/r2_gaussian](https://github.com/i-am-mushfiq/r2_gaussian) | [Ruyi-Zha/r2_gaussian](https://github.com/Ruyi-Zha/r2_gaussian) |
| **gaussian-splatting** | 3DGS baseline reference | [i-am-mushfiq/gaussian-splatting](https://github.com/i-am-mushfiq/gaussian-splatting) | [graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) |

Both repos follow the convention: `origin` = your fork, `upstream` = original author's repo.
To pull future upstream improvements: `git fetch upstream && git rebase upstream/main`

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Architecture](#2-repository-architecture)
3. [System Pipeline](#3-system-pipeline)
4. [Technical Architecture](#4-technical-architecture)
5. [Research Assumptions](#5-research-assumptions)
6. [Training System](#6-training-system)
7. [Data Pipeline](#7-data-pipeline)
8. [Script-by-Script Documentation](#8-script-by-script-documentation)
9. [Experiment Tracking](#9-experiment-tracking)
10. [Reproducibility Status](#10-reproducibility-status)
11. [Technical Debt](#11-technical-debt)
12. [Known Bugs and Risks](#12-known-bugs-and-risks)
13. [Environment Setup](#13-environment-setup)
14. [Immediate Priorities](#14-immediate-priorities)
15. [Research Opportunities](#15-research-opportunities)

---

## 1. Project Overview

### What This Project Does

This repository implements and applies **R²-Gaussian (Rectifying Radiative Gaussian Splatting for Tomographic Reconstruction)** — a NeurIPS 2024 method (Ruyi Zha et al.) — to CT reconstruction from a custom brain CT dataset. The central premise: replace the classical volumetric reconstruction pipeline (FDK, SART, iterative CT) with a differentiable scene representation using 3D Gaussian primitives trained end-to-end on sparse X-ray projections.

### Core Research Goal

Given a small set of X-ray projections (75 cone-beam projections over 360°) of a human brain, reconstruct a full 3D volumetric CT density field by optimizing a set of 3D Gaussian primitives whose simulated X-ray projections match the observed data.

### Scientific Problem

Classical CT reconstruction requires hundreds to thousands of projections (dose-intensive). Sparse-view CT reconstruction is an ill-posed inverse problem: you must recover a 3D volumetric density from 2D projection integrals at limited angles. The goal is to leverage the implicit regularization of a Gaussian scene representation to produce reconstructions that match or exceed classical algorithms at dramatically lower view counts.

### Why Gaussian Splatting?

3D Gaussian Splatting (3DGS) offers:
- Explicit, interpretable scene representation (no implicit MLP)
- Fast differentiable rendering via GPU tile-based rasterization
- Adaptive density control (densification/pruning) to concentrate detail
- Directly optimizable via gradient descent end-to-end

### Why Adapting 3DGS to CT Is Hard

1. **Rendering model change**: Standard 3DGS renders via alpha-compositing along viewing rays for RGB. X-ray rendering is a line integral of attenuation (Beer-Lambert), not a view-dependent RGB accumulation. The rendering equation is fundamentally different.

2. **No spherical harmonics**: CT density is view-independent — a Gaussian has scalar density, not directional color. The SH stage must be fully removed.

3. **Projection modes**: CT scanners use either cone-beam (perspective) or parallel-beam geometry. The projection matrix must be derived from physical scanner parameters (DSD, DSO, detector size), not from photographic camera calibration.

4. **Coordinate alignment**: TIGRE (the CT simulation toolkit) uses one coordinate convention; the CUDA rasterizer inherits 3DGS conventions (GLM column-major, transposed R). Scene coordinates must be reconciled across three systems: TIGRE, PyTorch, and the CUDA rasterizer.

5. **Volume querying**: Evaluation requires not just rendering projections but also extracting a full 3D volumetric density grid from the Gaussian cloud — a 3D voxelization pass. This required a second custom CUDA kernel (`GaussianVoxelizer`) that does not exist in standard 3DGS.

6. **Scale and numerical stability**: Scanner parameters (DSD, DSO) are in physical units (meters or mm). The entire scene is re-normalized to `[-1,1]³` before training to avoid numerical instability in the CUDA EWA splatting math.

---

## 2. Repository Architecture

```
Thesis_Defence/                         # Local parent directory (not a GitHub repo)
├── gaussian-splatting/                 # Forked from graphdeco-inria — 3DGS baseline (minimally modified)
│   ├── .gitignore                      # Excludes Data/, viewers/, build/, output/, colmap_database.db, spec.txt
│   ├── Data/Brain2_training/           # Same brain dataset run through standard 3DGS
│   ├── train.py                        # Standard 3DGS train loop
│   ├── scene/                          # 3DGS scene module
│   ├── gaussian_renderer/              # Standard RGB rasterizer
│   └── ...
├── r2_gaussian/                        # Forked from Ruyi-Zha — MAIN PROJECT — R²-Gaussian CT reconstruction
│   ├── train.py                        # ← Primary entry point
│   ├── test.py                         # Evaluation script
│   ├── initialize_pcd.py               # Point cloud initialization
│   ├── png_stack_to_npy.py             # Preprocessing: PNG stack → .npy volume
│   ├── r2_gaussian/                    # Core library package
│   │   ├── arguments/__init__.py       # ModelParams, OptimizationParams, PipelineParams
│   │   ├── dataset/
│   │   │   ├── __init__.py             # Scene class
│   │   │   ├── cameras.py              # Camera, MiniCam
│   │   │   └── dataset_readers.py      # readBlenderInfo, readNAFInfo, angle2pose
│   │   ├── gaussian/
│   │   │   ├── gaussian_model.py       # GaussianModel class (the core representation)
│   │   │   ├── render_query.py         # render() and query() — forward pass wrappers
│   │   │   └── initialize.py           # initialize_gaussian()
│   │   ├── utils/
│   │   │   ├── argument_utils.py
│   │   │   ├── camera_utils.py
│   │   │   ├── cfg_utils.py
│   │   │   ├── ct_utils.py             # TIGRE wrapper: recon_volume, run_ct_recon_algs
│   │   │   ├── gaussian_utils.py       # Math: build_rotation, get_expon_lr_func
│   │   │   ├── general_utils.py        # t2a (tensor→array), safe_state
│   │   │   ├── graphics_utils.py       # getWorld2View2, getProjectionMatrix
│   │   │   ├── image_utils.py          # metric_vol, metric_proj, psnr, ssim
│   │   │   ├── log_utils.py            # TensorboardX logger
│   │   │   ├── loss_utils.py           # l1_loss, ssim, tv_3d_loss
│   │   │   ├── plot_utils.py           # show_gaussians, show_two_volume, marching cubes viz
│   │   │   └── system_utils.py         # mkdir_p, searchForMaxIteration
│   │   └── submodules/
│   │       ├── simple-knn/             # CUDA KNN for initial distance computation
│   │       └── xray-gaussian-rasterization-voxelization/   # ← CRITICAL CUSTOM CUDA KERNEL
│   │           ├── cuda_rasterizer/    # 2D X-ray projection (forward+backward)
│   │           ├── cuda_voxelizer/     # 3D volume querying (forward+backward)
│   │           ├── ext.cpp             # PyTorch extension binding
│   │           └── setup.py
│   ├── data_generator/
│   │   ├── synthetic_dataset/
│   │   │   ├── generate_data.py        # Single-case data generation with TIGRE
│   │   │   ├── generate_data_all.py    # Batch generation
│   │   │   ├── process_raw_data.py     # Raw CT → normalized 256³ .npy
│   │   │   └── cone_ntrain_75_angle_360/Brain2_PNG_cone/   # ← ACTIVE TRAINING DATA
│   │   └── real_dataset/
│   │       ├── generate_data.py
│   │       └── generate_data_all.py
│   ├── scripts/
│   │   ├── train_all.py                # Batch training across cases
│   │   ├── visualize_scene.py          # Open3D: cameras + vol mesh
│   │   ├── ours_to_naf_format.py       # Convert our format to NAF pickle
│   │   ├── run_traditional_methods.py  # FDK, SART, CGLS, ASD-POCS baselines
│   │   └── train_all_saxnerf.py
│   ├── 3D_vis/                         # Visualization suite (25+ scripts)
│   │   ├── 3D_vis.py                   # Generic: pickle → voxel grid → marching cubes
│   │   ├── 3D_vis_chest.py, 3D_vis_head.py, ...  # Per-anatomy variants
│   │   └── create_gif_*.py             # Animated rotation GIFs
│   ├── brain/                          # Brain-specific visualization notebooks
│   ├── output/                         # Training outputs (gitignored)
│   │   ├── Brain2_training/            # 30k iter run — complete
│   │   ├── Brain2_training_v2/         # 60k iter run — abandoned at ~5k
│   │   └── [UUID-named]/              # Multiple early experimental runs
│   ├── dummy_renderer.py               # Debug: pickle → voxel grid → mesh render
│   ├── dummy_point_renderer.py         # Debug: pickle → Open3D point cloud
│   ├── dummy_init_pcd.py               # Alternate init logic (non-standard)
│   ├── dummy_TIGRE_run.py              # TIGRE installation sanity check
│   ├── PLYcomparer.py                  # Compare two PLY files
│   ├── pickle_to_ply.py                # Pickle → PLY converter v1
│   ├── pickle_to_ply_v2.py             # Pickle → PLY converter v2
│   ├── pickle_to_ply_v3.py             # Pickle → PLY converter v3 (canonical)
│   ├── ASCII_to_bin.py                 # ASCII PLY → binary PLY
│   ├── png_stack_to_npy.py             # PNG stack → normalized .npy volume
│   └── environment.yml                 # Conda environment specification
└── simple-knn/                         # Standalone clone of bkerbl/simple-knn (GitLab) — no local changes, not forked
```

### Module Responsibilities

| Module | Responsibility |
|--------|---------------|
| `r2_gaussian.gaussian.GaussianModel` | Scene representation — stores and optimizes the Gaussian cloud |
| `r2_gaussian.gaussian.render` | X-ray projection rendering (2D rasterization) |
| `r2_gaussian.gaussian.query` | 3D volume extraction (voxelization) |
| `r2_gaussian.dataset.Scene` | Data loading, camera management, bounding box |
| `r2_gaussian.dataset.dataset_readers` | Blender-format and NAF-format parsers |
| `cuda_rasterizer` | Tile-based X-ray 2D projection CUDA kernel |
| `cuda_voxelizer` | 3D volume accumulation CUDA kernel |
| `ct_utils.py` | TIGRE bridge: forward/backward CT algorithms |
| `initialize_pcd.py` | FDK-bootstrapped initial point cloud generation |

### Dependency Graph

```
train.py
  └── Scene (dataset/__init__.py)
        ├── dataset_readers.py → meta_data.json / .pickle
        └── camera_utils.py → Camera (cameras.py) → graphics_utils.py
  └── GaussianModel (gaussian/gaussian_model.py)
        └── simple_knn._C (CUDA KNN)
        └── gaussian_utils.py
  └── render() (gaussian/render_query.py)
        └── xray_gaussian_rasterization_voxelization.GaussianRasterizer (CUDA)
  └── query() (gaussian/render_query.py)
        └── xray_gaussian_rasterization_voxelization.GaussianVoxelizer (CUDA)
  └── loss_utils.py [l1_loss, ssim, tv_3d_loss]
  └── image_utils.py [metric_vol, metric_proj]
  └── log_utils.py → TensorboardX
```

---

## 3. System Pipeline

### 3.1 Data Ingestion

**Input**: PNG stack of brain CT slices (`data_generator/synthetic_dataset/Brain2_PNG/`)

**Step 1 — PNG → .npy**:
```
png_stack_to_npy.py <png_folder> <out.npy> [target_size=256]
```
Loads sorted PNG stack, converts to grayscale (luminosity), normalizes to [0,1] via percentile clipping (1–99%), resamples to isotropic cube of `target_size³`. Output: `float32` NumPy array.

**Step 2 — .npy → projection dataset**:
```
data_generator/synthetic_dataset/generate_data.py \
  --vol data_generator/synthetic_dataset/volume_processed/Brain2_PNG.npy \
  --scanner data_generator/synthetic_dataset/scanner/cone_beam.yml \
  --output data/synthetic_dataset/cone_ntrain_75_angle_360 \
  --n_train 75 --n_test 100
```
Uses TIGRE to perform forward projection (`tigre.Ax`) at `n_train` evenly-spaced angles over 360°. Adds Poisson noise (10000 photons) and Gaussian noise (σ=10). Produces:
- `proj_train/*.npy` — 75 projection images (512×512)
- `proj_test/*.npy` — 100 test projections
- `vol_gt.npy` — ground truth volume
- `meta_data.json` — scanner config + projection metadata

### 3.2 Preprocessing / Coordinate Transformation

**Scene scaling** (in `dataset_readers.py:readBlenderInfo`):
```python
scene_scale = 2 / max(meta_data["scanner"]["sVoxel"])
```
All physical quantities (`sVoxel`, `sDetector`, `DSD`, `DSO`, `offOrigin`, `offDetector`, `dVoxel`, `dDetector`) are multiplied by `scene_scale`. This maps the volume of interest into the unit cube `[-1,1]³`.

**Camera pose derivation** (`dataset_readers.py:angle2pose`):
Camera-to-world transform at angle `θ` is:
```
R = R3(θ) @ R2(+90°) @ R1(-90°)
T = [DSO·cos(θ), DSO·sin(θ), 0]
```
where `DSO` is the scaled distance from X-ray source to scanner origin.

**Projection image scaling**:
```python
image = np.load(image_path) * scene_scale
```
Projection intensities are also scaled by `scene_scale` for consistency.

**World-to-camera** (`graphics_utils.py:getWorld2View2`):
Standard `w2c = inv(c2w)`. R is stored transposed in the Gaussian model to match GLM conventions in the CUDA code.

**Projection matrix** (`graphics_utils.py:getProjectionMatrix`):
- Parallel beam (mode=0): Identity matrix `torch.eye(4)` — no perspective divide needed
- Cone beam (mode=1): Standard perspective projection with `znear=0.01, zfar=100.0`

**Note on dDetector convention**: `dDetector` is stored as `[v, u]` not `[u, v]`. FovX is computed from detector dimension `[1]` and FovY from dimension `[0]`. This is an implicit convention that must not be broken.

### 3.3 Initialization

```
python initialize_pcd.py --data <path> [--recon_method fdk] [--n_points 50000]
```

1. Load training projections and scanner geometry
2. Run FDK reconstruction via TIGRE
3. Threshold FDK volume at `density_thresh=0.05`
4. Sample `n_points=50000` positions from high-density voxels
5. Record voxel density values, rescale by `density_rescale=0.15`
6. Save as `init_<name>.npy` — shape `(N, 4)`: `[x, y, z, density]`

The `density_rescale=0.15` is empirical: raw FDK density values overestimate Gaussian density after integration along rays. The rescaling avoids immediate saturation.

### 3.4 Training Pipeline

```
python train.py -s <data_path> -m <output_path> [--config <yaml>]
```

Per-iteration:
1. Pop a random camera from the shuffled training stack (replenish when exhausted)
2. Call `render(viewpoint_cam, gaussians, pipe)` → rendered X-ray projection
3. Compare to GT projection with `l1_loss` + optional `dssim`
4. Optionally compute 3D TV loss on a random 32³ sub-volume
5. `loss.backward()`
6. Track gradient norms in `xyz_gradient_accum`
7. Adaptive control: densify (clone/split) + prune at intervals from iter 500–15000
8. `optimizer.step()`
9. Log to TensorboardX, save checkpoints at specified iterations

### 3.5 Evaluation Pipeline

During training (at `test_iterations`):
- Render all train+test cameras → compute per-projection PSNR/SSIM (2D)
- Query full volume with `query()` → compare to `vol_gt.npy` (3D PSNR/SSIM)
- Save to `eval/iter_XXXXXX/eval2d_*.yml` and `eval3d.yml`

Post-training (`test.py`):
- Load trained Gaussian pickle
- Re-evaluate render + reconstruction
- Save `.nii.gz` files for 3D Slicer visualization
- Save `.npy` volumes

---

## 4. Technical Architecture

### 4.1 Gaussian Representation

Each Gaussian has 4 trainable attributes:
| Attribute | Raw param | Activation | Shape |
|-----------|-----------|------------|-------|
| Position | `_xyz` | identity | `(N, 3)` |
| Density | `_density` | `Softplus` | `(N, 1)` |
| Scale | `_scaling` | `sigmoid * (smax−smin) + smin` or `exp` | `(N, 3)` |
| Rotation | `_rotation` | L2 normalize (unit quaternion) | `(N, 4)` |

**Critical**: Scale is bounded to `[scale_min, scale_max]` as a fraction of volume size. In the Brain2 experiment: `scale_min=0.0005`, `scale_max=0.5` (of 2.0 units = `[0.001, 1.0]` world units).

**No spherical harmonics**: Standard 3DGS uses view-dependent SH color. Here, density is scalar and view-independent. The SH code in `forward.cu` is retained from the original codebase but disabled with `#! We dont need this.` comments.

### 4.2 X-ray Rendering (GaussianRasterizer)

The custom CUDA rasterizer in `cuda_rasterizer/` implements:

**Preprocessing** (per Gaussian):
1. Transform Gaussian centers to view space: `t = viewmatrix @ xyz`
2. Compute 2D covariance via EWA splatting (Zwicker et al. 2002, Eq. 29/31):
   - `cov2D = J @ (T_view_matrix @ cov3D @ T_view_matrix.T) @ J.T`
   - For parallel beam: `J = diag(focal_x, focal_y, 1)` (identity-like)
   - For cone beam: standard perspective Jacobian
3. Compute conic parameters `(a, b, c, density)` from 2D covariance
4. Compute 2D screen radius; mark tiles touched
5. Sort Gaussians by (tile_id | depth)

**Rendering** (per tile, per pixel):
- Accumulate: `rendered = Σ density_i * exp(-0.5 * r^T conic r) * mu_i`
- `mu_i` is the Gaussian's projected mean (screen position)
- This is a **line-integral approximation** — each Gaussian contributes to a pixel based on the overlap of its 2D projection with that pixel
- The sum is **additive** (not alpha-composited), modeling X-ray transmission

**Single channel**: `NUM_CHANNELS=1`. Rendered image is a scalar attenuation map per pixel.

### 4.3 3D Volume Querying (GaussianVoxelizer)

The voxelizer in `cuda_voxelizer/` accumulates Gaussian density into a 3D grid:
- Block size: `8×8×8` voxels per thread block
- Parameters: `nVoxel_x/y/z` (grid resolution), `sVoxel_x/y/z` (physical size), `center_x/y/z`
- For each voxel, accumulate contributions from all overlapping Gaussians
- Differentiable: backward pass propagates gradients from volume metrics back to Gaussian params

This is the mechanism by which 3D PSNR/SSIM is computed (and through TV loss, also trained).

### 4.4 Optimization

**Optimizer**: Adam with per-parameter learning rate groups (no weight decay)

**Learning rate schedule**: Exponential decay per parameter group:
| Param | LR init | LR final | Steps |
|-------|---------|---------|-------|
| xyz | 0.00016 (Brain2 custom) | 1.6e-6 | 30000 |
| density | 0.01 | 0.001 | 30000 |
| scaling | 0.005 | 0.0005 | 30000 |
| rotation | 0.001 | 0.0001 | 30000 |

**Note**: The Brain2 position LR (0.00016) differs from the default (0.0002). Position LR is multiplied by `spatial_lr_scale=1.0` (no scaling applied).

**Loss**:
```
L_total = L1(render, gt) + λ_dssim * (1 - SSIM(render, gt)) + λ_tv * TV_3D(sub_vol)
```
- λ_dssim = 0.2, λ_tv = 0.05 (Brain2 config)
- TV is computed on a randomly sampled 32³ voxel sub-region (not the full volume)

### 4.5 Adaptive Density Control

**IMPORTANT**: In the Brain2 experiments, density control is **effectively disabled**:
```yaml
densify_from_iter: 35000
densify_until_iter: 35000
```
Both are beyond the 30000-iteration budget, so no densification occurs. The Gaussian count remains fixed at the initialization count (50000 from FDK sampling).

When enabled (default settings):
- Clone: Gaussians with gradient norm ≥ threshold AND scale ≤ `densify_scale_threshold` are duplicated with halved density
- Split: Gaussians with gradient norm ≥ threshold AND scale > threshold are split into 2 with scale/0.8
- Prune: Remove Gaussians with density < `density_min_threshold` OR outside bbox OR too large on screen

---

## 5. Research Assumptions

### Mathematical
- X-ray projection follows Beer-Lambert line integral: `I = Σ μ_i * δs_i` (attenuation sum along ray)
- The Gaussian splatting approximation models this via 2D projected Gaussian footprints
- This is an approximation — not physically exact for thick Gaussians relative to their depth extent
- Scale bounds `[0.001, 1.0]` in world units are empirically chosen; no principled derivation

### Imaging
- CT projections are linear attenuation maps (no beam hardening correction)
- Projections are normalized to `[0,1]` range
- Scene scale `2/max(sVoxel)` is applied to projections too — intensity calibration is implicit
- Noise model: Poisson (I₀=10000) + Gaussian (σ=10) is a rough approximation of real CT noise

### Coordinate-Space
- Scanner uses **right-handed** convention; CUDA uses column-major (GLM) with R transposed
- The `dDetector` convention is `[v, u]` (row, column) — not `[u, v]`
- All coordinates normalized to `[-1,1]³` — absolute physical scale is lost during training
- `offOrigin=[0,0,0]` assumed for all Brain2 scans (scanner centered on object)

### Reconstruction
- Ground truth volume is the same Brain2 PNG stack resampled to 256³
- Evaluation assumes perfect registration between ground truth and reconstructed volumes
- The voxelization query uses the same `nVoxel/sVoxel` as the scanner — exact alignment assumed

### Physics
- No scattering model
- No polychromatic X-ray spectrum effects
- No detector point spread function

---

## 6. Training System

### Training Loop Location
`r2_gaussian/train.py:training()` — see lines 34–220

### Losses (documented)
| Loss | Weight | Description |
|------|--------|-------------|
| L1 | 1.0 | Per-pixel L1 on rendered vs GT projection |
| dSSIM | λ=0.2 | `1 - SSIM` on rendered vs GT projection |
| TV 3D | λ=0.05 | Total variation on randomly sampled 32³ sub-volume |

### Optimizer
- Adam (eps=1e-15), one parameter group per attribute
- `optimizer.zero_grad(set_to_none=True)` used for memory efficiency

### Schedulers
Exponential decay: `lr(t) = exp(log(lr_init)*(1-t) + log(lr_final)*t)` where `t = step/max_steps`. Implemented as a closure in `gaussian_utils.py:get_expon_lr_func`.

### Checkpoints
- Saved as `.pth` (PyTorch pickle) at `checkpoint_iterations`
- Gaussian model state saved as `.pickle` (not `.ply`) at `saving_iterations`
- The pickle format stores raw pre-activation parameters (`_xyz`, `_density`, `_scaling`, `_rotation`, `scale_bound`)
- **WARNING**: The model pickle is NOT a standard PLY — it cannot be loaded by PLY viewers directly. See `pickle_to_ply_v3.py` for conversion.

### Convergence Assumptions
- 30000 iterations on a 256³ volume with 50000 initial Gaussians and no densification converged on Brain2 data
- Training is seeded: `random.seed(0)`, `np.random.seed(0)`, `torch.manual_seed(0)`
- `shuffle=False` in `Scene.__init__` for the Brain2 experiment — cameras are not shuffled

### Distributed Training
Not implemented. Single GPU only. Device hardcoded to `cuda:0` in `general_utils.py:safe_state`.

---

## 7. Data Pipeline

### Dataset Structure (Active)
```
r2_gaussian/data_generator/synthetic_dataset/cone_ntrain_75_angle_360/Brain2_PNG_cone/
├── vol_gt.npy                          # Ground truth 256³ float32 CT volume
├── init_Brain2_PNG_cone.npy            # Initial point cloud (N×4: x,y,z,density)
├── meta_data.json                      # Scanner config + projection file paths
├── proj_train/                         # 75 × (512×512) float32 .npy projections
│   ├── proj_train_0000.npy
│   └── ... (0000–0074)
└── proj_test/                          # 100 × (512×512) float32 .npy projections
    └── proj_test_0000.npy ... (0000–0099)
```

### Scanner Configuration (meta_data.json)
```json
{
  "mode": "cone",
  "DSD": 7.0,           // Distance Source Detector (meters, pre-scaling)
  "DSO": 5.0,           // Distance Source Origin (meters, pre-scaling)
  "nDetector": [512, 512],
  "sDetector": [4.0, 4.0],
  "nVoxel": [256, 256, 256],
  "sVoxel": [2.0, 2.0, 2.0],  // Physical volume size in meters
  "totalAngle": 360.0,
  "startAngle": 0.0,
  "noise": true,
  "possion_noise": 10000,      // Typo in source: should be "poisson_noise"
  "gaussian_noise": [0, 10]
}
```

### Normalization
1. Volume normalized to [0,1] in `process_raw_data.py` (percentile normalization)
2. Scene scaled to [-1,1]³: `scene_scale = 2.0 / 2.0 = 1.0` (since `max(sVoxel) = 2.0`)
3. Projection intensities scaled by `scene_scale` (= 1.0 in this case)
4. No data augmentation applied

### NAF Format
For compatibility with SAX-NeRF, data can be converted to NAF `.pickle` format via `scripts/ours_to_naf_format.py`. Key difference: NAF stores measurements in mm (×1000), our format in meters.

### Real Dataset Path
`data_generator/real_dataset/` — contains `generate_data.py` and `generate_data_all.py` for real X-ray data. Does not appear to have been used yet with the brain dataset.

---

## 8. Script-by-Script Documentation

### `r2_gaussian/train.py`
**Purpose**: Main training entry point  
**Inputs**: `--source_path`, `--model_path`, optional `--config <yaml>`  
**Outputs**: `output/<name>/point_cloud/iteration_*/point_cloud.pickle`, `cfg_args.yml`, TensorBoard events, `eval/` YML files  
**Dependencies**: All of `r2_gaussian.*`, CUDA extensions  
**Execution**: `python train.py -s <data_path> -m <output_path>`  
**Hidden assumptions**: Must be run from `r2_gaussian/` directory (sys.path manipulation). CUDA device 0 hardcoded.  
**Failure points**: Missing `init_*.npy`; missing CUDA extension builds; device OOM if too many Gaussians  

### `r2_gaussian/test.py`
**Purpose**: Post-hoc evaluation of a trained model  
**Inputs**: `--model_path`, optional `--iteration` (default -1 = latest), flags to skip render/recon  
**Outputs**: `test/iter_*/` with `.npy`, `.png`, `.nii.gz`, and `.yml` eval files  
**Side effects**: Saves `.nii.gz` for 3D Slicer  
**Execution**: `python test.py -m output/Brain2_training`  
**Hidden assumptions**: Reads `cfg_args` from model_path to recover source_path  

### `r2_gaussian/initialize_pcd.py`
**Purpose**: Generate initial point cloud from FDK reconstruction  
**Inputs**: `--data <path>`, `--output <path>`, `--recon_method fdk|random`, `--n_points 50000`  
**Outputs**: `init_<name>.npy` — shape `(N, 4)`  
**Dependencies**: TIGRE  
**Hidden assumptions**: Asserts `valid_voxels >= n_points`. Will crash if volume is very sparse.  
**Note**: `density_rescale=0.15` is empirical and may need tuning for new data  

### `r2_gaussian/png_stack_to_npy.py`
**Purpose**: Convert PNG slice stack to normalized .npy volume  
**Inputs**: `<png_folder>`, `<out.npy>`, optional `[target_size]`, `[spacing_z spacing_y spacing_x]`  
**Outputs**: Float32 .npy volume at `target_size³`  
**Dependencies**: skimage, scipy  
**Note**: Uses percentile normalization (1–99%). Grayscale via luminosity formula. This is a custom preprocessing script not part of the original R²-Gaussian repo.  

### `r2_gaussian/scripts/visualize_scene.py`
**Purpose**: 3D visualization of scene geometry (cameras + volume mesh)  
**Inputs**: `-s <data_path>` + optional `--mc_thresh`, `--cam_scale`, `--upsample_factor`, `--subdivide`  
**Outputs**: Opens Open3D window  
**Dependencies**: Open3D, scipy (zoom for upsample)  
**Note**: Requires `vol_gt.npy` to exist in source path  

### `r2_gaussian/scripts/run_traditional_methods.py`
**Purpose**: Run FDK, SART, CGLS, ASD-POCS, OS-ASD-POCS baselines for comparison  
**Inputs**: Path to NAF `.pickle` data  
**Outputs**: Per-method reconstructions with PSNR/SSIM in `eval_3d.yml`  
**Dependencies**: TIGRE  

### `r2_gaussian/3D_vis/3D_vis.py`
**Purpose**: Visualize trained Gaussian cloud as 3D mesh  
**Inputs**: Hardcoded path to `output/Brain2_training/point_cloud/iteration_30000/point_cloud.pickle`  
**Outputs**: Opens matplotlib 3D interactive window  
**Side effects**: None  
**WARNING**: Path is hardcoded. Must be modified for different experiments.  

### `r2_gaussian/dummy_renderer.py`
**Purpose**: Alternate 3D mesh visualization with multi-angle saves  
**Inputs**: Same hardcoded pickle path  
**Outputs**: PNG images at `brain/elevation_*/angle_*.png`  
**Note**: This is a debugging/thesis-demo script, not production code  

### `r2_gaussian/pickle_to_ply_v3.py`
**Purpose**: Convert trained Gaussian pickle to PLY for viewers  
**Inputs**: Hardcoded paths to `output/Brain2_training/...`  
**Outputs**: `point_cloud.ply` (ASCII)  
**Note**: The bounding box normalization in this script (`points_normalized = bbox_min + points * scale`) is WRONG — it treats raw `_xyz` (pre-activation, in `[-1,1]³`) as if it needs re-normalization. The xyz values are already in world coordinates.  

### `r2_gaussian/dummy_init_pcd.py`
**Purpose**: Alternative initialization with 50th-percentile density threshold  
**Inputs**: Hardcoded path to `Brain2_PNG_cone/vol_gt.npy`  
**Note**: Uses different `dVoxel=[2,2,2]` and `offOrigin=[0,0,0]` rather than reading from scanner config. More fragile than `initialize_pcd.py`.  

### `r2_gaussian/data_generator/synthetic_dataset/generate_data.py`
**Purpose**: Forward-project a ground-truth volume to synthetic X-ray projections  
**Inputs**: `--vol`, `--scanner <yaml>`, `--output`, `--n_train`, `--n_test`  
**Outputs**: Full NeRF-format dataset directory with `meta_data.json`  
**Dependencies**: TIGRE, plotly (only for visualization, can skip)  

---

## 9. Experiment Tracking

### Active Experiments

#### `output/Brain2_training/` — Primary Result
- **Config**: 30000 iterations, 75-view cone beam, Brain2_PNG, densification disabled
- **Position LR**: 0.00016 (custom, 80% of default 0.0002)
- **Max Gaussians**: 200000
- **Status**: Complete
- **Final metrics** (iter 30000):
  - `psnr_3d: 6.22`, `ssim_3d: 0.014`
  - `psnr_2d: 11.00`, `ssim_2d: 0.557`
- **Assessment**: 3D metrics are extremely poor. `psnr_3d=6.22` is barely better than noise; `ssim_3d=0.014` indicates near-zero structural similarity. The 2D projection metrics are mediocre but functional. This result likely reflects a suboptimal initialization or scanner parameter mismatch, not a code bug.

#### `output/Brain2_training_v2/` — Abandoned
- **Config**: 60000 iterations, same data, densification re-enabled (5000–40000), higher save/eval frequency
- **Status**: Abandoned after ~5000 iterations (only `iter_000001` through `iter_005000` eval dirs exist)
- **Metrics at iter 5000**: identical to Brain2_training final (`psnr_3d: 6.22, ssim_3d: 0.014`) — suggesting metrics were not improving

#### `output/[UUID-named]/` — Early Experiments (8 runs)
- Multiple runs with auto-generated UUIDs from early training exploration
- No labeled configs; these predate the named experiments

### Dead Ends / Abandoned Directions
1. **Higher iteration count (60k)**: Abandoned — no improvement visible at 5k iters in v2
2. **Alternative pickle-to-PLY conversions**: Three versions suggest difficulty getting viewer-compatible output (PLYcomparer.py hardcodes a nonexistent `room/input.ply` path)
3. **Densification**: Disabled in Brain2 runs because the gradient-based densification did not improve quality in initial tests

### Promising Signals
- 2D projection SSIM of 0.557 shows the model is fitting the projection data to some degree
- The model does converge (loss decreases monotonically as shown in TensorBoard event files)
- The Brain2_PNG_cone data is correctly formatted and loads without errors

### Incomplete Work
- Real dataset pipeline (`data_generator/real_dataset/`) — scripts exist but no real X-ray data present
- No traditional method baseline comparison has been run against the Brain2 data
- No ablation on number of training views (25/50/75)
- No hyperparameter tuning for the Brain2 scanner geometry
- `Brain2_training_v2` was likely abandoned without systematic diagnosis

---

## 10. Reproducibility Status

### Current Status: PARTIALLY REPRODUCIBLE

| Component | Status | Notes |
|-----------|--------|-------|
| Environment | Reproducible | `environment.yml` exists |
| CUDA extension build | Fragile | Requires CUDA 11.8 + matching PyTorch; TIGRE build is manual |
| Data (Brain2_PNG) | Present | In `data_generator/synthetic_dataset/Brain2_PNG/` |
| Generated dataset | Present | `cone_ntrain_75_angle_360/Brain2_PNG_cone/` |
| Init point cloud | Present | `init_Brain2_PNG_cone.npy` exists |
| Trained model | Present | `output/Brain2_training/point_cloud/iteration_30000/point_cloud.pickle` |
| Training config | Present | `output/Brain2_training/cfg_args.yml` |
| Eval results | Present | All YAML files present |

### Missing / Risk Points
1. **Hardcoded absolute paths in `cfg_args`**: `source_path` contains `C:\Users\Mushfiq\Desktop\Thesis Defence\...`. On a new machine, the path will not exist. Must override with `-s` flag.
2. **TIGRE manual build**: TIGRE v2.3 must be built from source for the GPU. `dummy_TIGRE_run.py` exists as an install check.
3. **xray-gaussian-rasterization-voxelization**: Must be compiled via `pip install -e .` from `r2_gaussian/r2_gaussian/submodules/xray-gaussian-rasterization-voxelization/`. CUDA toolkit and PyTorch versions must match exactly.
4. **simple-knn**: Also must be compiled: `pip install -e .` from `r2_gaussian/r2_gaussian/submodules/simple-knn/` (and separately from `simple-knn/` at the repo root).
5. **`debug=True` in Brain2 config**: The training config has `debug: true`. This enables extra CUDA assertions and significantly slows training. Should be `false` for production.
6. **Platform sensitivity**: `sys.path.append("./")` patterns assume execution from project root. The Windows path conventions in cfg_args confirm development on Windows.

---

## 11. Technical Debt

### Hardcoded Paths (Critical)
- `dummy_renderer.py:21` — hardcoded pickle path to `output/Brain2_training/...`
- `dummy_point_renderer.py:7` — same
- `dummy_init_pcd.py:6-7` — hardcoded vol_gt and output paths
- `pickle_to_ply_v3.py:15-17` — hardcoded output/Brain2_training paths
- `ASCII_to_bin.py:6-8` — same
- `PLYcomparer.py:101-102` — hardcodes both a Brain2 output and a nonexistent `gaussian-splatting/output/room` path
- `3D_vis/3D_vis.py:28` — hardcoded pickle path
- All `3D_vis/3D_vis_*.py` scripts — various hardcoded category paths

### Duplicated Logic
- Three `pickle_to_ply.py` versions (v1, v2, v3) — the progression is iterative debugging, no cleanup
- Two separate `simple-knn` installations: `simple-knn/` at root AND `r2_gaussian/r2_gaussian/submodules/simple-knn/`
- `dummy_init_pcd.py` reimplements initialization logic from `initialize_pcd.py` with different (incorrect) parameters

### Broken Code
- `PLYcomparer.py:102` — references `gaussian-splatting/output/room/input.ply` which does not exist
- `forward.cu` — contains full SH implementation (`computeColorFromSH`) that is entirely dead code; marked `! We dont need this`
- `pickle_to_ply_v3.py` — the normalization logic (`points_normalized = bbox_min + points * scale`) is semantically incorrect for the already-scaled xyz coordinates

### Missing Modularity
- All `3D_vis_*.py` scripts are near-identical with minor parameter changes (object name, threshold, paths) — should be a single configurable script
- `dummy_*` scripts should be removed or moved to a `debug/` folder

### Hidden Coupling
- `train.py` hardcodes `torch.cuda.set_device(torch.device("cuda:0"))` — fails on machines without GPU 0 or with different GPU assignments
- `dataset_readers.py` has implicit assumption that `dDetector[0]` = v-dimension, `dDetector[1]` = u-dimension — breaks if data is prepared differently
- `cfg_args` file in output uses `eval` to parse a Namespace string — arbitrary code execution risk

### Scaling Risks
- No multi-GPU support
- `plot_utils.py:show_gaussians()` loops over Gaussians in Python for visualization — O(N) with expensive Open3D mesh creation; unusable for >10k Gaussians

---

## 12. Known Bugs and Risks

### High Severity
1. **Poor 3D reconstruction quality**: `psnr_3d=6.22, ssim_3d=0.014`. Not a code bug per se, but indicates the experiment configuration is wrong. Likely causes:
   - Scanner parameters (DSD=7, DSO=5) may not match the scale of the Brain2 volume
   - Initialization quality from FDK on this specific dataset may be poor
   - Densification being disabled means the 50k initial Gaussians may be insufficient
2. **debug=True in production**: Causes ~2–5× slowdown in CUDA kernel due to assertions

### Medium Severity
3. **v2 run abandonment at iter 5000**: Densification was re-enabled in v2 but metrics at iter 5000 equal the *final* metrics of v1 (iter 30000) — suggests either densification is removing Gaussians too aggressively, or metrics are cached
4. **Missing scanner YAML in tracked files**: The `scanner/cone_beam.yml` file used for data generation is not in the Git history (the data directory is gitignored with `data_generator/synthetic_dataset/Brain2_PNG/`)

### Low Severity
5. **`possion_noise` typo in meta_data.json**: Field name misspelled; code reads it correctly but any schema validation would fail
6. **SSIM pixel-max normalization in metric_proj**: Slices are individually max-normalized before SSIM computation — this removes absolute intensity information and may inflate SSIM scores
7. **Thread pool in test.py multithread_write**: Uses `max_workers=None` (unbounded pool), which can exhaust OS file handles for large datasets

---

## 13. Environment Setup

### Conda Environment
```bash
git clone https://github.com/i-am-mushfiq/r2_gaussian.git --recursive
cd r2_gaussian
conda env create --file environment.yml
conda activate r2_gaussian_new
```

Key dependencies:
- Python 3.9
- CUDA 11.8 (system must match)
- numpy 1.24.1 (pinned — later versions break CUDA extensions)
- open3d 0.18.0 (for visualization)
- pytorch/torchvision: not in yml — must install separately matching CUDA 11.8
- SimpleITK (for .nii.gz output in test.py)

### CUDA Extension Builds
```bash
# From r2_gaussian/
SET DISTUTILS_USE_SDK=1  # Windows only

# 1. Install X-ray rasterizer/voxelizer
pip install -e r2_gaussian/submodules/xray-gaussian-rasterization-voxelization --no-build-isolation

# 2. Install simple-knn  
pip install -e r2_gaussian/submodules/simple-knn --no-build-isolation
```

### TIGRE Installation
```bash
# Download TIGRE v2.3
wget https://github.com/CERN/TIGRE/archive/refs/tags/v2.3.zip
unzip v2.3.zip
pip install TIGRE-2.3/Python --no-build-isolation
```
Test with: `python r2_gaussian/dummy_TIGRE_run.py`

### PyTorch (not in yml)
```bash
pip install torch==2.0.1+cu118 torchvision==0.15.2+cu118 --index-url https://download.pytorch.org/whl/cu118
```

### CUDA Compiler
- NVCC from CUDA 11.8 toolkit required
- On Windows: `SET DISTUTILS_USE_SDK=1` must be set before building
- Tested on Ubuntu 20.04 + RTX 3090 (from original R²-Gaussian README)
- Development machine appears to be Windows 11 (from path conventions in cfg_args)

### GPU Requirements
- Minimum: RTX 3090 (24GB) recommended for 256³ volume
- `cuda:0` hardcoded — must be the training GPU
- Approximate memory: 8–16GB for 50k–200k Gaussians at 256³ volume

---

## 14. Immediate Priorities for New Team

### Fix First
1. **Remove `debug: true`** from any training config before running experiments. Likely 2–3× speedup.
2. **Verify scanner parameters match the Brain2 volume scale**. Run `scripts/visualize_scene.py` and check that camera frustums actually enclose the reconstructed mesh.
3. **Fix `pickle_to_ply_v3.py` normalization bug** (line 76–77) — the xyz are already in world coords, no renormalization needed.

### Validate First
1. **Run `dummy_TIGRE_run.py`** on the new machine — confirms TIGRE is installed and GPU-accessible
2. **Run `initialize_pcd.py --evaluate`** — prints `3D PSNR for initial Gaussians`. If this is very low (<15 dB), the FDK init is failing and you should try `--recon_method random` or adjust `density_thresh`/`density_rescale`
3. **Reproduce Brain2_training results** to confirm environment parity before making changes

### Reproduce First
1. The complete training pipeline: `initialize_pcd.py` → `train.py` → `test.py`
2. Verify `psnr_3d` and `ssim_3d` match the logged values in `output/Brain2_training/eval/iter_030000/eval3d.yml`

### Refactor First
1. **Consolidate `3D_vis_*.py`** into one script with `--subject` argument
2. **Remove duplicate pickle_to_ply** versions (keep v3 as canonical)
3. **Eliminate dummy_* scripts** or move to `debug/`

### Investigate First
1. **Why does `ssim_3d=0.014`?** Possible causes: (a) volume coordinate mismatch between gt and prediction, (b) density scale mismatch, (c) poor FDK initialization. Start by visualizing `vol_gt.npy` vs `vol_pred.npy` using `show_two_volume()` from `plot_utils.py`.
2. **What happens with densification enabled?** Run v3 experiment with `densify_from_iter=500, densify_until_iter=15000, densify_grad_threshold=5e-5` (paper defaults)

---

## 15. Research Opportunities

### Immediate Improvements (Technical)
1. **Scanner parameter calibration**: The Brain2 PNG data may have a different physical scale than the default scanner YAML. Calibrating DSD/DSO/sVoxel to match the actual acquisition geometry could dramatically improve reconstruction quality.
2. **Initialization tuning**: Try `--density_thresh=0.1`, `--density_rescale=0.3` or use the random initialization mode for the Brain2 dataset.
3. **Enable densification**: The paper shows densification is critical. Running with default settings (`densify_from_iter=500, densify_until_iter=15000`) on Brain2 should be the next experiment.

### Research Extensions
1. **Sparse-view experiments**: Generate 25-view and 10-view datasets, compare R²-Gaussian vs FDK/SART. This is the core comparison the thesis likely needs.
2. **Real CT data**: The `data_generator/real_dataset/` infrastructure exists but hasn't been used with the brain data. Applying to actual DICOM brain scans (post-conversion) would be a strong contribution.
3. **Traditional baseline comparison**: `scripts/run_traditional_methods.py` implements FDK, SART, CGLS, ASD-POCS. Running these on Brain2 would establish the baseline against which R²-Gaussian should be compared.
4. **Volume resolution scaling**: The current setup uses 256³. Testing 128³ vs 512³ would characterize the method's scaling behavior.
5. **Parallel vs cone beam**: The code supports both. A comparison on the same dataset would be novel.

### Paper Opportunities
1. **Brain CT-specific reconstruction analysis**: The existing 3D_vis suite generates compelling visualizations — these can support a paper on Gaussian splatting for neuroimaging CT
2. **Ablation study**: lambda_tv, lambda_dssim, n_training_views, initialization method — all configurable, none systematically evaluated
3. **Comparison with NeRF-based methods**: SAX-NeRF data format is already supported. A direct comparison on NAF-format brain data is well within reach.

### Evaluation Improvements
1. Add **RMSE** and **SSIM per-slice-axis** to the standard output — these provide more granular diagnostic information
2. Add **FID** (Fréchet Inception Distance) or **perceptual metrics** for visual quality assessment beyond PSNR/SSIM
3. **Timing benchmarks**: The original paper reports 5–15 min training on RTX 3090 — logging iteration time and total training time for Brain2 would contextualize the results
