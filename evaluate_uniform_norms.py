import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

def test_func(x, y):
    return np.sin(x * y) + 1.0 / (1.0 + np.exp(-100*(x - y)))

def calculate_norms_from_csv(csv_filename):
    print(f"Loading grid data from '{csv_filename}'...")
    
    # 1. Load the flat table
    df = pd.read_csv(csv_filename)
    
    # 2. Extract the unique X and Y axis coordinate points
    # This automatically finds the original 1D x_pts and y_pts
    x_pts = np.unique(df['X_Coordinate'].values)
    y_pts = np.unique(df['Y_Coordinate'].values)
    
    res_x = len(x_pts)
    res_y = len(y_pts)
    print(f"Detected uniform grid resolution: {res_x}x{res_y} ({len(df):,} total rows)")
    
    # 3. Reshape the flat function column back into a 2D matrix
    # Crucial: Using 'C' or 'F' ordering depends on how it was flattened. 
    # Since your original meshgrid used indexing='ij', it reshapes cleanly like this:
    func_grid_2d = df['Function_Value'].values.reshape(res_x, res_y)
    
    # 4. Reconstruct the Uniform Grid Interpolator
    uniform_interp = RegularGridInterpolator((x_pts, y_pts), func_grid_2d, method='linear')
    
    # 5. Create a Dense Disjoint Test Set (as per the paper's methodology)
    # We choose 200x200 points slightly offset so we evaluate *between* the table nodes.
    x_test = np.linspace(-1.99, 1.99, 200)
    y_test = np.linspace(-1.99, 1.99, 200)
    X_test, Y_test = np.meshgrid(x_test, y_test, indexing='ij')
    test_points = np.vstack([X_test.flatten(), Y_test.flatten()]).T
    
    # 6. Compute exact true values vs interpolated values at test points
    print("Evaluating interpolator against dense test set...")
    true_vals = test_func(test_points[:, 0], test_points[:, 1])
    interp_vals = uniform_interp(test_points)
    
    # 7. Calculate Pointwise Absolute Error
    rel_errors = np.abs((interp_vals - true_vals))
        
    # 8. Compute global error norms
    l1_norm = np.mean(rel_errors)                         # Mean Absolute Error
    l2_norm = np.sqrt(np.mean(rel_errors**2))             # Root Mean Square Error
    linf_norm = np.max(rel_errors)                        # Maximum Absolute Error (Worst Case)
    
    # Display Results
    print("\n================ ERROR NORMS ================")
    print(f"L1 Norm:    {l1_norm:.6f}")
    print(f"L2 Norm:    {l2_norm:.6f}")
    print(f"L_inf Norm: {linf_norm:.6f}")
    print("=============================================\n")
    
    return l1_norm, l2_norm, linf_norm

# --- Execute Script ---
if __name__ == "__main__":
    # Point this to the CSV file you want to evaluate
    target_csv = 'tables/uniform_grid_func_evals/uniform_evaluations-128.csv'
    
    try:
        calculate_norms_from_csv(target_csv)
    except FileNotFoundError:
        print(f"Error: Could not find '{target_csv}'. Make sure the file exists in this directory.")
