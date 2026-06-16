# Quadtree Surrogate From Uniform Tables

## Overview
This repository builds an adaptive quadtree surrogate from a uniform data table and evaluates interpolation error/size tradeoffs across saved tree files.

The current workflow does not require knowing the original analytical function. Instead, tree refinement is driven by values in a uniform CSV table with columns `X`, `Y`, and `F`.

## Repository Workflow
1. Generate or provide a uniform grid table in CSV format (`X`, `Y`, `F`).
2. Build an adaptive quadtree from that table and save it as `.npz`.
3. Load one or more saved tree files and compute full-domain $L_1$, $L_2$, and $L_\infty$ norms.
4. Save norm/size summaries as CSV and plots.

## Main Scripts
- `build_quadtree.py`
	- Loads a uniform source table (`SOURCE_TABLE_FILE`).
	- Builds adaptive quadtree cells using cubic local interpolation and a user-specified error threshold.
	- Saves:
		- tree archive: `tables/quadtree-{MAX_DEPTH}-{ERROR_THRESHOLD}-{resolution}.npz`
		- grid visualization: `figs/quadtree_grid.png`

- `load_and_use_table.py`
	- Loads one or more saved quadtree `.npz` files.
	- Evaluates full-domain norms on a structured test grid based on a user-specified resolution and assumption of the original function.
	- Saves:
		- comparison CSV: `tables/quadtree_norm_size_comparison.csv`
		- scatter plot: `figs/quadtree_norm_vs_size.png`
		- bar plot: `figs/quadtree_norms_by_file.png`

- `build_uniform_table.py`
	- Generates a uniform table CSV under `tables/uniform_grid_func_evals/`.
	- Useful for creating source data when you do not already have a table during testing. 

## Legacy / Older Utilities
- `evaluate_norms.py`: Older norm-evaluation utility based on direct CSV interpolation.

These are retained for reference but are not the primary pipeline.

## Expected Data Format
Uniform source table CSV must contain:
- `X`: x-coordinate
- `Y`: y-coordinate
- `F`: sampled function/value at `(X, Y)`

The table should represent a rectangular uniform grid.

## Quick Start
1. Build or refresh a source uniform table:
	 - `python build_uniform_table.py`
2. Build quadtree from the table:
	 - `python build_quadtree.py`
3. Evaluate saved trees and generate comparison artifacts:
	 - `python load_and_use_table.py`

## Notes
- Domain bounds, error thresholds, depth limits, and file lists are configured directly in each script.
- The original function is not required for building the tree, but is needed for norm evaluation. The current test function is a 2D tanh, which is smooth and has interesting curvature.
  - The hope is that by choosing a smoother function the error threshold will always be the bottleneck rather than the max depth, allowing for more meaningful comparisons across error thresholds.
- This will need to be converted to C++ at some point. 
- Equation of State tables are the motivating application. This will likely require some additional handling for non-rectangular domains and non-uniform source tables.
  - After EOS tables, the next step will be to apply this to radiation transport tables. 
  - Hemholtz EOS table is the first target, following the paper by J Carlson, S Couch, and B O'Shea. 
- Eventual comparison to a NN surrogate is planned.
- The hope is that the memory savings are significant enough to allow the allow the surrogate to run on a GPU. 
- 
