import numpy as np
import os
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CloughTocher2DInterpolator

def test_func(x, y):
    return np.sin(x * y) + 1.0 / (1.0 + np.exp(-100*(x - y)))

def calculate_quadtree_norms(csv_filename):
    test_resolution = 500
    domain_xmin, domain_xmax = 0, 1
    domain_ymin, domain_ymax = 0, 1
    x_test_global = np.linspace(domain_xmin, domain_xmax, test_resolution)
    y_test_global = np.linspace(domain_ymin, domain_ymax, test_resolution)
    
    # Load the offline point cloud
    df = pd.read_csv(csv_filename)
    points = df[['X', 'Y']].values
    values = df['F'].values
    
    # Build the smooth cubic interpolator (Instantiate this once)
    # This handles the complex geometric triangulation under the hood
    interp_func = CloughTocher2DInterpolator(points, values)

    i = 0
    err = np.empty(test_resolution*test_resolution)
    # Iterate through each point to compute error 
    for x_query in x_test_global:
        for y_query in y_test_global:
            interp_val = interp_func(x_query, y_query)
            true_val = test_func(x_query, y_query)

            err[i] = np.abs(interp_val - true_val)
            i = i + 1

    # 3. Calculate Mathematical Norms across all cell centers
    l1_norm = np.mean(err)                           # L1: Mean Absolute Error at centers
    l2_norm = np.sqrt(np.mean(err**2))               # L2: Root Mean Square Error at centers
    linf_norm = np.max(err)                          # L_inf: Worst-case Maximum Error at centers

    # Print clean summary table
    print("\n========= QUADTREE NORMS =========")
    print(f"L1 Norm:    {l1_norm:.6f}")
    print(f"L2 Norm:    {l2_norm:.6f}")
    print(f"L_inf Norm: {linf_norm:.6f}")
    print("====================================")

    return l1_norm, l2_norm, linf_norm

if __name__ == "__main__":
    # Path to your points data file
    csv_file = "tables/quadtree_points_CT-7-0.01-500.csv" 
    l1_norm, l2_norm, linf_norm = calculate_quadtree_norms(csv_file)
    
    table_size = os.path.getsize(csv_file) / 1024
    print(f"Table Size (Data Footprint): {table_size} KB")
    print(f"L_inf norm: {linf_norm}")
