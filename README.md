# Adaptive Log-Space Quadtree Surrogate for High-Z Tables

An efficient surrogate model builder for tabulated data spanning many orders of magnitude.

Important: Must run from the quadtree root directory, not the `python/` subdirectory. An example command is:
```bash
python -m python.build_quadtree
```

## Overview

This project builds an adaptive quadtree from dense HDF5 tables and evaluates surrogate accuracy by comparing quadtree interpolation against reference data.

The key innovation is log-space refinement, which naturally resolves multi-decade physics data more efficiently than linear-space trees.

## Problem Context

The training and testing data have:
- Density (den): ranges from ~10^-12 to ~10^15
- Temperature (temp): ranges from ~10^3 to ~10^13
- Function values (f): span ~10^49 in magnitude

A linear-space tree fails badly on this data. Log-space refinement allocates cells adaptively to match the actual variation scale.

## Data Format

### HDF5 File Structure
```
training_data.hdf5 / testing_data.hdf5
├── den                   # (2400, 6480) density grid
├── temp                  # (2400, 6480) temperature grid
└── Table_Values
    └── f                 # (2400, 6480) target function values
```

- **den** varies along columns; all rows contain the same x-coordinates.
- **temp** varies along rows; all columns contain the same y-coordinates.
- **f** is sampled on the den-temp grid.

This is a regular grid in (den, temp) space. Coordinates are always positive (required for log10).

## Main Scripts

### `build_quadtree.py`
Builds an adaptive quadtree surrogate in log10-space.

Workflow:
1. Load training data from `training_data.hdf5`.
2. Convert den and temp to log10.
3. Fit a global bicubic interpolating spline (`RectBivariateSpline`) to g = symlog(f, `value_linthresh`) on the log-space grid; it supplies g, g_x, g_y, g_xy at cell corners.
4. Refine the domain pass by pass, splitting every leaf whose bicubic Hermite patch exceeds the error threshold on the table points it covers. After each pass the corner data is reassigned with hanging-node constraints (see Interpolators), and all leaves are re-checked.
5. Save tree to `tables/quadtree-{MAX_DEPTH}-{ERROR_THRESHOLD}-{RESOLUTION}-hermite.h5`.

Key parameters (in code):
- `max_depth`: Maximum tree depth (default: 6). Increase to 7–8 for finer resolution.
- `error_threshold`: Relative error stopping criterion (default: 1e-4).
- `value_linthresh`: Linear threshold of the symlog transform the leaves interpolate in (default: 1e21). See Interpolators.
- `hdf5_file`: Path to training data.

Output:
- HDF5 file (`quadtree_v3`) of flat arrays:
  - `is_leaf` (uint8, one per node in preorder, children in SW, SE, NW, NE order). Every internal node has four children, so these flags define the whole tree; cell bounds and split points are recomputed from the root `bounds` attribute on load, bit-identical to the build.
  - `coefficients` (n_leaves × 4 × 4 float64), the leaves' Hermite matrices in the same preorder.
  - Attributes: `format`, `interpolant`, `coord_space`, `value_linthresh`, root `bounds`.
- File size is essentially the coefficient payload (128 bytes per leaf).

### `calc_error.py`
**Loads a quadtree and evaluates error on testing data.**

Workflow:
1. Load testing data from `testing_data.hdf5`.
2. Build a RegularGridInterpolator on the log-space test grid.
3. Load a saved quadtree and evaluate it at random test points.
4. Compute error metrics (absolute and relative).
5. Print norms.

Key parameters (in code):
- `tree_file`: Path to `.npz` quadtree file to evaluate.
- `n_random_points`: Number of random points to sample (default: 100,000).
- `random_seed`: RNG seed for reproducibility.

Output:
- Console output: L_1, L_2, L_infty relative error norms and file size.

## Architecture

### QuadTreeNode
- Stores a rectangular cell in log10-space.
- Holds a 4×4 Hermite matrix at leaf nodes: g, g_x, g_y, g_xy at the four corners, where g = symlog(f, `value_linthresh`) and derivatives are w.r.t. the log10 coordinates.
- `evaluate` returns f = inverse_symlog(g, `value_linthresh`); the linthresh is stored in the tree file.
- Navigates children by quadrant: NE, NW, SE, SW.
- `evaluate(x, y)` converts physical (den, temp) to log10 automatically and traverses the tree.

### Interpolators
- **Build phase**: global bicubic spline of g = symlog(f, `value_linthresh`) on the log10-space grid provides corner values and derivatives.
- **Value space**: f spans ~10^49 and changes sign along a curve in the (den, temp) plane, where |f| is still ~10^13 one grid point away. With a small linthresh, symlog(f) jumps by ~2·log10|f| across that curve, and cubic overshoot there turns into astronomically wrong f after `inverse_symlog`. `value_linthresh` must therefore exceed |f| near the sign change. 1e21 (just above the largest positive f, ~3.7e20) makes the transform linear through the sign change and logarithmic in the large negative region, which gave the fewest leaves for a given error. For this table, linthresh ≲ 1e17 blows up near the sign change; linthresh → ∞ reduces to interpolating raw f.
- **Leaf interpolant**: tensor-product cubic Hermite (bicubic) patch. Neighbouring leaves share corner data, so g, and hence f, and their gradients are continuous (C1) across every cell edge. A corner that lies inside a coarser neighbour's edge (a hanging node) takes its value and derivatives from that neighbour's patch rather than the spline, which keeps the surface C1 where cell sizes change.
- **C++ module**: `polyinterp.evaluate_hermite()` evaluates a leaf's Hermite patch at (x, y) within the cell's bounds. Rebuild it after changing `cpp/polyinterp.cpp` with `python setup.py build_ext --inplace` from `cpp/`.
- Trees saved with the earlier cubic Lagrange interpolant are rejected by `load_quadtree`; rebuild them.

### Coordinate Space
Trees can be built in:
- `log10`: Log10-space.
- `linear`: Linear space.

The `coord_space` metadata is saved in the `.npz` file, so loaders auto-detect the correct evaluation path.

## Quick Start

### 1. Build the Quadtree
```bash
python build_quadtree.py
```
Output: `tables/quadtree-{MAX_DEPTH}-{ERROR_THRESHOLD}-{RESOLUTION}.npz`

### 2. Evaluate Error
```bash
python calc_error.py
```
Output: L_1, L_2, L_infty error norms on random test points.

### 3. Adjust Parameters
Edit `build_quadtree.py`:
```python
max_depth = 7  # Finer tree
error_threshold = 1e-5  # Stricter convergence
```
Then rebuild: `python build_quadtree.py` and re-evaluate.

## File Organization
```
.
├── README.md
├── build_quadtree.py       # Builder
├── calc_error.py           # Error evaluator
├── cpp/
│   ├── polyinterp.cpp      # C++ polynomial evaluator
│   └── setup.py
├── training_data.hdf5      # Input: training domain
├── testing_data.hdf5       # Input: evaluation domain
└── tables/
    └── quadtree-*.npz      # Output: saved trees
```
Note: `training_data.hdf5` and `testing_data.hdf5` are not included in the repository; they must be provided separately.
