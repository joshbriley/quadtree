import numpy as np
import os
import pandas as pd
from scipy.interpolate import RegularGridInterpolator
from scipy.interpolate import CloughTocher2DInterpolator

def test_func(x, y):
    return np.sin(x * y) + 1.0 / (1.0 + np.exp(-100*(x - y)))

def calculate_norms(csv_filename):
    test_resolution = 256
    domain_xmin, domain_xmax = -2, 2
    domain_ymin, domain_ymax = -2, 2
    x_test_global = np.linspace(domain_xmin + 0.01, domain_xmax, test_resolution)
    y_test_global = np.linspace(domain_ymin + 0.01, domain_ymax, test_resolution)
    
    # Load the CSV
    df = pd.read_csv(csv_filename)
    points = df[['X', 'Y']].values
    values = df['F'].values
    
    # Build Clough-Tocher Interpolator on global table
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

    # Calculate Norms 
    l1_norm = np.mean(err) 
    l2_norm = np.sqrt(np.mean(err**2))
    linf_norm = np.max(err)

    # Print clean summary table
    print("\n========= NORMS =========")
    print(f"L1 Norm:    {l1_norm:.6f}")
    print(f"L2 Norm:    {l2_norm:.6f}")
    print(f"L_inf Norm: {linf_norm:.6f}")
    print("===========================")

    return l1_norm, l2_norm, linf_norm

if __name__ == "__main__":
    # Path to CSV data file
    csv_file = "tables/quadtree_points_CT-15-0.01-500.csv" 
    # csv_file = "tables/uniform_grid_func_evals/uniform_evaluations-128.csv" 
    l1_norm, l2_norm, linf_norm = calculate_norms(csv_file)
    
    table_size = os.path.getsize(csv_file) / 1000 # 1000 or 1024?
    print(f"Table Size: {table_size} kB")
    print("---------------------------")
