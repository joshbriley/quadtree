import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle


def get_leaf_cells(npz_file):
    data = np.load(npz_file)

    bounds = data["bounds"]
    is_leaf = data["is_leaf"]
    children = data["child_indices"]

    leaf_cells = []
    stack = [(0, 0)]  # (node index, depth)

    while stack:
        i, d = stack.pop()
        xmin, xmax, ymin, ymax = bounds[i]

        if is_leaf[i]:
            leaf_cells.append((xmin, xmax, ymin, ymax))
        else:
            for c in children[i]:
                if c >= 0:
                    stack.append((int(c), d + 1))

    return leaf_cells


def plot_quadtree(cells, output="figs/quadtree-0.01.png"):
    fig, ax = plt.subplots(figsize=(8, 6))

    for xmin, xmax, ymin, ymax in cells:
        ax.add_patch(Rectangle(
            (xmin, ymin),
            xmax - xmin,
            ymax - ymin,
            fill=False,
            edgecolor="black",
            linewidth=0.5
        ))

    cells = np.array(cells)
    ax.set_xlim(cells[:, 0].min(), cells[:, 1].max())
    ax.set_ylim(cells[:, 2].min(), cells[:, 3].max())

    ax.set_xlabel("log10(Density)")
    ax.set_ylabel("log10(Temperature)")
    ax.set_title("Quadtree Decomposition")
    ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(output, dpi=300)
    plt.close()


if __name__ == "__main__":
    file = "tables/quadtree-10-0.01-4321.npz"

    cells = get_leaf_cells(file)
    plot_quadtree(cells)

    print(f"{len(cells)} leaf cells plotted.")