import pandas as pd
import matplotlib.pyplot as plt
import os

file_paths = [
    "2.csv", "4.csv", "8.csv", "16.csv", "20.csv",
    "24.csv", "32.csv", "64.csv", "128.csv", "256.csv",
    "512.csv", "1024.csv", "2048.csv"
]

distinct_dark_colors = [
    "#FFE927", "#FD0101", "#00FF26", "#7300FFAE", "#FF53F1",
    "#000000", "#A9A9A9", "#04726FFF", "#0004FFFF", "#068100",
    "#5FFFDF", "#66693B", "#FF8400FF"
]

plt.figure(figsize=(16, 9))

all_matrix_sizes = set()

# Each file = one tile size curve
for idx, path in enumerate(file_paths):
    tile_size = int(os.path.basename(path).replace(".csv", ""))
    df = pd.read_csv(path)

    # Compute MPKI for this CSV
    df["MPKI"] = (df["L1-dcache-load-misses"] / df["L1-dcache-loads"]) * 1000

    matrix_sizes = df["Matrix size"]
    mpki_vals = df["MPKI"]

    all_matrix_sizes.update(matrix_sizes.tolist())

    plt.plot(matrix_sizes, mpki_vals, marker="o", linestyle="-",
             color=distinct_dark_colors[idx % len(distinct_dark_colors)],
             label=f"Tile {tile_size}", linewidth=2, markersize=5)

plt.xlabel("Matrix Size")
plt.ylabel("MPKI")
plt.title("MPKI vs Matrix Size for Different Tile Sizes")
plt.legend(title="Tile Size", bbox_to_anchor=(1.05, 1),
           loc='upper left', fontsize=7, ncol=2)

# Use log scale on X (matrix sizes are powers of 2)
plt.xscale("log")
plt.xticks(sorted(all_matrix_sizes), labels=sorted(all_matrix_sizes), rotation=45)

plt.grid(True, which="major", axis="y")  # horizontal grid lines
plt.tight_layout()
plt.savefig("mpki_vs_matrix_tile.png", dpi=300, bbox_inches="tight")
plt.show()
