import argparse
import time
import numpy as np
import matplotlib.pyplot as plt

'''
python driver.py \
  --csv tables/uniform_grid_func_evals/uniform_evaluations-128.csv \
  --thresholds 1e-1 1e-2 1e-3 1e-4 1e-5
'''

from build_quadtree import (
    load_uniform_table,
    build_quadtree,
    save_quadtree,
)

from calc_error import (
    calculate_norms_for_quadtree,
    test_func,
)


def run_driver(csv_path, max_depth, thresholds, test_resolution):
    """
    Build quadtrees for multiple thresholds and compute error norms.
    """

    # ----------------------------------------------------------
    # Load training data
    # ----------------------------------------------------------
    print("Loading uniform grid data...")
    (
        x_coords,
        y_coords,
        global_points,
        global_vals,
        source_interpolator,
    ) = load_uniform_table(csv_path)

    xmin, xmax = float(x_coords.min()), float(x_coords.max())
    ymin, ymax = float(y_coords.min()), float(y_coords.max())
    resolution = len(x_coords)

    # ----------------------------------------------------------
    # Build test grid (for error computation)
    # ----------------------------------------------------------
    print(f"Creating test grid: {test_resolution} x {test_resolution}")
    x_test = np.linspace(xmin, xmax, test_resolution)
    y_test = np.linspace(ymin, ymax, test_resolution)

    X_test, Y_test = np.meshgrid(x_test, y_test, indexing="ij")
    true_vals = test_func(X_test, Y_test)

    results = []

    # ----------------------------------------------------------
    # Main loop over thresholds
    # ----------------------------------------------------------
    for thresh in thresholds:
        print("\n" + "=" * 60)
        print(f"[RUN] threshold={thresh:.2e}, max_depth={max_depth}")

        # ------------------------------------------------------
        # Build quadtree
        # ------------------------------------------------------
        t0 = time.time()

        root = build_quadtree(
            xmin,
            xmax,
            ymin,
            ymax,
            thresh,
            max_depth,
            global_points,
            global_vals,
            source_interpolator,
        )

        build_time = time.time() - t0
        print(f"Build time: {build_time:.2f} sec")

        # ------------------------------------------------------
        # Save quadtree
        # ------------------------------------------------------
        output_file = (
            f"tables/quadtree-{max_depth}-{thresh:.1e}-{resolution}.npz"
        )

        save_quadtree(root, output_file)

        # ------------------------------------------------------
        # Compute error (using your calc_error module)
        # ------------------------------------------------------
        print("Computing error norms...")

        t1 = time.time()

        l1, l2, linf, size_kb = calculate_norms_for_quadtree(
            output_file,
            x_test,
            y_test,
            true_vals,
        )

        err_time = time.time() - t1

        print(f"Error time: {err_time:.2f} sec")
        print(f"L1   : {l1:.4e}")
        print(f"L2   : {l2:.4e}")
        print(f"Linf : {linf:.4e}")
        print(f"Size : {size_kb:.2f} kB")

    

        # ------------------------------------------------------
        # Store results
        # ------------------------------------------------------
        results.append({
            "threshold": thresh,
            "file": output_file,
            "build_time": build_time,
            "error_time": err_time,
            "L1": l1,
            "L2": l2,
            "Linf": linf,
            "size_kb": size_kb,
        })

    return results


# ----------------------------------------------------------
# CLI interface
# ----------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Quadtree driver: build + save + error analysis"
    )

    parser.add_argument(
        "--csv",
        type=str,
        required=True,
        help="Path to uniform function evaluation CSV",
    )

    # parser.add_argument(
    #     "--max_depth",
    #     type=int,
    #     required=True,
    #     help="Maximum quadtree depth",
    # )

    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        required=True,
        help="List of error thresholds",
    )

    parser.add_argument(
        "--test_resolution",
        type=int,
        default=300,
        help="Resolution of test grid for error computation",
    )

    args = parser.parse_args()

    input_resolution = 128  # This should match the resolution used in build_uniform_table.py
    min_points_per_cell = 3  # This should match the value used in build_qu
    max_depth = int(np.log(input_resolution/min_points_per_cell) / np.log(2))  # Max depth based on minimum points per cell and initial resolution

    results = run_driver(
        args.csv,
        max_depth,
        args.thresholds,
        args.test_resolution,
    )

    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    for r in results:
        print(
            f"thresh={r['threshold']:.2e} | "
            f"Linf={r['Linf']:.3e} | "
            f"L2={r['L2']:.3e} | "
            f"L1={r['L1']:.3e} | "
            f"size={r['size_kb']:.2f}kB | "
            f"build={r['build_time']:.2f}s | "
            f"file={r['file']}"
        )


    # Plot L_1, L_2, L_inf vs size_kb. Use log scale for y-axis. 
    sizes = [r["size_kb"] for r in results]
    l1_vals = [r["L1"] for r in results]
    l2_vals = [r["L2"] for r in results]
    linf_vals = [r["Linf"] for r in results]
    thresholds = [r["threshold"] for r in results]

    plt.figure(figsize=(10, 6))
    plt.scatter(sizes, l1_vals, label="L1", marker="o")
    plt.scatter(sizes, l2_vals, label="L2", marker="s")
    plt.scatter(sizes, linf_vals, label="L_inf", marker="^")
    plt.xscale("linear")
    plt.yscale("log")
    plt.xlabel("Table Size (kB)")
    plt.ylabel("Norm Value")
    plt.title("Norm vs Table Size for Different Thresholds")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/norm_vs_size.png")
    plt.close()

    # Plot L_1, L_2, L_inf vs threshold.
    plt.figure(figsize=(10, 6))
    plt.scatter(thresholds, l1_vals, label="L1", marker="o")
    plt.scatter(thresholds, l2_vals, label="L2", marker="s")
    plt.scatter(thresholds, linf_vals, label="L_inf", marker="^")
    plt.xscale("log")
    plt.yscale("log")
    plt.xlabel("Error Threshold")
    plt.ylabel("Norm Value")
    plt.title("Norm vs Error Threshold")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig("figs/norm_vs_threshold.png")
    plt.close()

    
    
