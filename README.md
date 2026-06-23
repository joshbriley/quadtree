# Adaptive Log-Space Quadtree Surrogate for High-Z Tables

An efficient surrogate model builder for tabulated data spanning many orders of magnitude.

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
3. Build a RegularGridInterpolator on the log-space grid.
4. Recursively partition the domain, refining cells where local cubic interpolation error exceeds threshold.
5. Save tree to `tables/quadtree-{MAX_DEPTH}-{ERROR_THRESHOLD}-{RESOLUTION}.npz`.

Key parameters (in code):
- `max_depth`: Maximum tree depth (default: 6). Increase to 7–8 for finer resolution.
- `error_threshold`: Relative error stopping criterion (default: 1e-4).
- `hdf5_file`: Path to training data.

Output:
- `.npz` file containing flattened tree structure, bounds, leaf coefficients, and metadata.

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
- Holds 4×4 nodal values (polynomial coefficients) at leaf nodes.
- Navigates children by quadrant: NE, NW, SE, SW.
- `evaluate(x, y)` converts physical (den, temp) to log10 automatically and traverses the tree.

### Interpolators
- **Build phase**: RegularGridInterpolator on log10-space grid (fast, no triangulation).
- **C++ module**: `polyinterp.evaluate_polynomial()` evaluates cubic Lagrange polynomials at (x, y) within a cell's bounds.

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

## Error Metrics

### Relative Error
Used during tree refinement to decide cell splitting:
$$\text{rel\_error} = \frac{|\text{pred} - \text{truth}|}{|\text{truth}| + \varepsilon}$$

This ensures cells are refined where they matter most relative to local signal magnitude.

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
