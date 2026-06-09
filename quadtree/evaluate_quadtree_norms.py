import numpy as np
import os
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

def test_func(x, y):
    return np.sin(x * y) + 1.0 / (1.0 + np.exp(-100*(x - y)))

def calculate_quadtree_norms(points_csv_filename):
    # 1. Automatically locate the structural boxes file to get cell definitions
    boxes_csv_filename = points_csv_filename.replace("quadtree_points_", "quadtree_boxes_")
    print(f"Loading quadtree leaf boxes from '{boxes_csv_filename}'...")
    
    if not os.path.exists(boxes_csv_filename):
        raise FileNotFoundError(
            f"Could not find the structural boxes file: '{boxes_csv_filename}'.\n"
            f"Make sure you have run your main script to generate both files."
        )
        
    df_boxes = pd.read_csv(boxes_csv_filename)
    
    # Convert rows directly into a list of tuples: (xmin, xmax, ymin, ymax, depth)
    boxes = list(df_boxes.itertuples(index=False, name=None))
    print(f"Loaded {len(boxes)} leaf nodes.")

    abs_errors = []

    # 2. Iterate through each leaf cell to compute error at its exact center
    for xmin, xmax, ymin, ymax, depth in boxes:
        # Calculate the precise center coordinate of this specific cell
        x_mid = (xmin + xmax) / 2.0
        y_mid = (ymin + ymax) / 2.0
        
        # Reconstruct the 4-corner mesh vertices for this leaf (2x2 grid)
        x_corners = np.array([xmin, xmax])
        y_corners = np.array([ymin, ymax])
        X_c, Y_c = np.meshgrid(x_corners, y_corners, indexing='ij')
        
        # Evaluate the anchored true values at these 4 corner points
        corner_vals = test_func(X_c, Y_c)

        # Build the local linear patch interpolator using ONLY the 4 corners
        local_interp = RegularGridInterpolator((x_corners, y_corners), corner_vals, method='linear')

        # Predict the value at the cell's center point
        center_query = np.array([[x_mid, y_mid]])
        interp_val = local_interp(center_query)[0]
        true_val = test_func(x_mid, y_mid)

        # Compute pointwise Absolute Error at the center
        err = np.abs(interp_val - true_val)
        abs_errors.append(err)

    abs_errors = np.array(abs_errors)

    # 3. Calculate Mathematical Norms across all cell centers
    l1_norm = np.mean(abs_errors)                           # L1: Mean Absolute Error at centers
    l2_norm = np.sqrt(np.mean(abs_errors**2))               # L2: Root Mean Square Error at centers
    linf_norm = np.max(abs_errors)                          # L_inf: Worst-case Maximum Error at centers

    # Print clean summary table
    print("\n========= QUADTREE NORMS (CELL CENTERS) =========")
    print(f"L1 Norm:    {l1_norm:.6f}")
    print(f"L2 Norm:    {l2_norm:.6f}")
    print(f"L_inf Norm: {linf_norm:.6f}")
    print("==================================================")

    return l1_norm, l2_norm, linf_norm

if __name__ == "__main__":
    # Path to your points data file
    csv_file = 'tables/quadtree_grid/quadtree_points_15_0.001.csv'
    
    l1_norm, l2_norm, linf_norm = calculate_quadtree_norms(csv_file)
    
    table_size = os.path.getsize(csv_file) / 1024
    print(f"Table Size (Data Footprint): {table_size} KB")
    print(f"L_inf norm: {linf_norm}")
