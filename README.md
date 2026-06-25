# Quadtree surrogate from uniform tables

This repository is a small Python/C++ experiment for building an adaptive 2D quadtree surrogate from a **uniform CSV table** of sampled values.

## What is in the current code

- `build_uniform_table.py` generates a uniform table on `[-2, 2] x [-2, 2]` for `tanh(x * y)` and writes `X`, `Y`, `F` values to `tables/uniform_grid_func_evals/`.
- `build_quadtree.py` loads one of those CSV tables, builds an adaptive quadtree, and saves it as a compressed `.npz` file in `tables/`.
- `calc_error.py` loads saved quadtree files and computes `L1`, `L2`, and `L_inf` errors on a test grid, then writes `tables/quadtree_norm_size_comparison.csv`.
- `driver.py` is the main end-to-end script for sweeping over multiple error thresholds and generating comparison plots in `figs/`.
- `cpp/` contains the `polyinterp` extension used by the Python quadtree code.

## Setup

The scripts depend on `numpy`, `pandas`, `scipy`, `matplotlib`, and `pybind11`.

Build/install the interpolation extension first:

```bash
python -m pip install -e ./cpp
```

## Expected input table

The quadtree builder expects a rectangular uniform CSV table with these columns:

- `X`
- `Y`
- `F`

## Typical workflow

1. Generate a source table:

```bash
python build_uniform_table.py
```

2. Build a single quadtree using the hardcoded settings in `build_quadtree.py`:

```bash
python build_quadtree.py
```

3. Evaluate saved quadtree files listed in `calc_error.py`:

```bash
python calc_error.py
```

4. Run the full threshold sweep:

```bash
python driver.py \
  --csv tables/uniform_grid_func_evals/uniform_evaluations-128.csv \
  --thresholds 1e-1 1e-2 1e-3 1e-4 1e-5
```

## Notes

- Most settings are still hardcoded in the scripts.
- The current evaluation path assumes the same analytic test function used to generate the table is known.
