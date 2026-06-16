import numpy as np
import os
import csv
import matplotlib.pyplot as plt
from quadtree_toy import load_quadtree, test_func

# --- Configuration ---
DOMAIN_XMIN, DOMAIN_XMAX = -2.0, 2.0
DOMAIN_YMIN, DOMAIN_YMAX = -2.0, 2.0
TEST_RESOLUTION = 300
QUADTREE_FILES = [
    "tables/quadtree-7-1e-05-256.npz", 
    "tables/quadtree-7-0.0001-256.npz",
    "tables/quadtree-7-0.001-256.npz",
    "tables/quadtree-7-0.01-256.npz",
    "tables/quadtree-7-0.1-256.npz",
    "tables/quadtree-9-1e-05-256.npz", 
    "tables/quadtree-9-0.0001-256.npz",
    "tables/quadtree-9-0.001-256.npz",
    "tables/quadtree-9-0.01-256.npz",
    "tables/quadtree-9-0.1-256.npz",
]

OUTPUT_CSV = "tables/quadtree_norm_size_comparison.csv"
OUTPUT_SCATTER_PNG = "figs/quadtree_norm_vs_size.png"
OUTPUT_BAR_PNG = "figs/quadtree_norms_by_file.png"


def calculate_norms_for_quadtree(quadtree_file, x_test, y_test, true_vals):
    loaded_quadtree = load_quadtree(quadtree_file)
    interp_vals = np.empty_like(true_vals, dtype=float)

    for i, x in enumerate(x_test):
        for j, y in enumerate(y_test):
            interp_vals[i, j] = loaded_quadtree.evaluate(x, y)

    abs_error = np.abs(interp_vals - true_vals)
    l1_norm = np.mean(abs_error)
    l2_norm = np.sqrt(np.mean(abs_error**2))
    linf_norm = np.max(abs_error)
    table_size_kb = os.path.getsize(quadtree_file) / 1000.0
    return l1_norm, l2_norm, linf_norm, table_size_kb


def main():
    x_test = np.linspace(DOMAIN_XMIN, DOMAIN_XMAX, TEST_RESOLUTION)
    y_test = np.linspace(DOMAIN_YMIN, DOMAIN_YMAX, TEST_RESOLUTION)
    X_test, Y_test = np.meshgrid(x_test, y_test, indexing="ij")
    true_vals = test_func(X_test, Y_test)

    print("\n--- Full-Domain Norms vs Table Size ---")
    print(f"Grid Resolution: {TEST_RESOLUTION} x {TEST_RESOLUTION}")
    print("file | size_kb | L1 | L2 | L_inf")
    print("-----|---------|----|----|------")

    results = []

    for quadtree_file in QUADTREE_FILES:
        # print(f"Evaluating {os.path.basename(quadtree_file)}...")
        l1_norm, l2_norm, linf_norm, table_size_kb = calculate_norms_for_quadtree(
            quadtree_file, x_test, y_test, true_vals
        )
        file_name = os.path.basename(quadtree_file)
        results.append({
            "file": file_name,
            "size_kb": table_size_kb,
            "L1": l1_norm,
            "L2": l2_norm,
            "L_inf": linf_norm,
        })
        print(
            f"{file_name} | "
            f"{table_size_kb:.2f} | "
            f"{l1_norm:.6e} | {l2_norm:.6e} | {linf_norm:.6e}"
        )

    os.makedirs("tables", exist_ok=True)
    os.makedirs("figs", exist_ok=True)

    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["file", "size_kb", "L1", "L2", "L_inf"])
        writer.writeheader()
        writer.writerows(results)

    sizes = np.array([r["size_kb"] for r in results])
    l1_vals = np.array([r["L1"] for r in results])
    l2_vals = np.array([r["L2"] for r in results])
    linf_vals = np.array([r["L_inf"] for r in results])
    labels = [r["file"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.scatter(sizes, l1_vals, label="L1", marker="o")
    ax.scatter(sizes, l2_vals, label="L2", marker="s")
    ax.scatter(sizes, linf_vals, label="L_inf", marker="^")
    ax.set_xlabel("Table Size (kB)")
    ax.set_ylabel("Norm Value")
    ax.set_yscale("log")
    ax.set_title("Norm vs Table Size")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_SCATTER_PNG, dpi=200)
    plt.close(fig)

    x_idx = np.arange(len(labels))
    width = 0.25
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.bar(x_idx - width, l1_vals, width, label="L1")
    ax.bar(x_idx, l2_vals, width, label="L2")
    ax.bar(x_idx + width, linf_vals, width, label="L_inf")
    ax.set_xticks(x_idx)
    ax.set_xticklabels(labels, rotation=45, ha="right")
    ax.set_yscale("log")
    ax.set_ylabel("Norm Value")
    ax.set_title("Norms by Quadtree Table")
    ax.grid(True, axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_BAR_PNG, dpi=200)
    plt.close(fig)

    print(f"Saved CSV: {OUTPUT_CSV}")
    print(f"Saved plot: {OUTPUT_SCATTER_PNG}")
    print(f"Saved plot: {OUTPUT_BAR_PNG}")
    print("---")


if __name__ == "__main__":
    main()
