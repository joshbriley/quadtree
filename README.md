# Quadtree decomposition

## Description
Decomposing a domain using the quadtree method and comparing the interpolation error to a uniform discretization.

Currently using the Clough-Tochard Interpolation method for training and testing. 

### Python Scripts
- `quadtree_toy.py`: Generates quadtree CSV file (`tables/quadtree_points_CT-{MAX_DEPTH}-{ERROR_THRESHOLD}-{TRAIN_RESOLUTION}.csv`) with x-points, y-points, and their cooresponding function evaluations.

- `test_func_eval.py`: Evaluates the test function on a uniform grid, plots, and exports a CSV with (`tables/uniform_evaluations-{resolution}.csv`) x-points, y-points, and their cooresponding function evaluations (same format as quadtree CSV)

- `evaluate_norms.py`: Evalues the $L_1$, $L_2$ and $L_\infty$ norms given a CSV file in which the first column is labeled 'X' (x-values), the second column is labeled 'Y' (y-values), and the third column is labled 'F' (function evalutions). 
