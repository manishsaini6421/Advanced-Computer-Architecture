import pandas as pd
import matplotlib.pyplot as plt
import os

# List of all 13 file paths
file_paths = [
    "2.csv", "4.csv", "8.csv", "16.csv", "20.csv",
    "24.csv", "32.csv", "64.csv", "128.csv", "256.csv",
    "512.csv", "1024.csv", "2048.csv"
]

# Define a new palette of 13 dark, highly distinguishable colors
distinct_dark_colors = [
    "#FFE927",  
    "#FD0101",  
    "#00FF26",  
    "#7300FFAE",  
    "#FF53F1",  
    "#000000",  
    "#A9A9A9",  
    "#04726FFF",  
    "#0004FFFF",  
    "#068100",  
    "#5FFFDF",  
    "#66693B",  
    "#FF8400FF"   
]

# Set up the figure
plt.figure(figsize=(16, 9))

# Collect unique matrix sizes for x-ticks
all_matrix_sizes = set()

# Plot each tile size separately with a unique dark color
for idx, path in enumerate(file_paths):
    # Extract tile size from filename
    tile_size = int(os.path.basename(path).replace(".csv", ""))
    
    # Read file
    df = pd.read_csv(path)
    
    # Calculate MPKI
    df["MPKI"] = (df["L1-dcache-load-misses"] / df["L1-dcache-loads"]) * 1000
    
    # Keep track of matrix sizes
    all_matrix_sizes.update(df["Matrix size"].tolist())
    
    # Plot MPKI vs Matrix size with a unique dark color
    plt.plot(df["Matrix size"], df["MPKI"], marker="o", linestyle="-", 
             color=distinct_dark_colors[idx], label=f"Tile {tile_size}",
             linewidth=2, markersize=5)

plt.xlabel("Matrix Size")
plt.ylabel("MPKI")
plt.title("MPKI vs Matrix Size for Different Tile Sizes")
plt.legend(title="Tile Size", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=7, ncol=2)
plt.grid(True, which="major", axis="y")  # Horizontal grid lines only

# Use logarithmic scale for symmetric spacing
plt.xscale("log")

# Set x-axis ticks to exact matrix sizes
plt.xticks(sorted(all_matrix_sizes), labels=sorted(all_matrix_sizes))

# Adjust layout to prevent legend clipping
plt.tight_layout()

# Save graph as PNG
plt.savefig("mpki_vs_matrix_tile_logscale.png", dpi=300, bbox_inches="tight")

plt.show()

print("✅ Graph saved as mpki_vs_matrix_tile_logscale.png")