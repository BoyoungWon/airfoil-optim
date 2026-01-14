import os
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend
import matplotlib.pyplot as plt

# ===================== CONFIG =====================

airfoil_dir = "airfoil_samples_1000"      # folder with .dat files
out_dir     = "airfoil_images_1000"       # folder to save PNGs
dpi         = 200

os.makedirs(out_dir, exist_ok=True)

# ===================== IO =====================

def load_dat(path):
    with open(path, "r") as f:
        lines = f.readlines()

    pts = []
    for ln in lines[1:]:  # skip name line
        parts = ln.strip().split()
        if len(parts) >= 2:
            pts.append([float(parts[0]), float(parts[1])])
    return np.array(pts, dtype=float)

# ===================== BATCH PLOT =====================

files = sorted(f for f in os.listdir(airfoil_dir) if f.endswith(".dat"))

print(f"Found {len(files)} airfoils")

for fname in files:
    path = os.path.join(airfoil_dir, fname)

    try:
        pts = load_dat(path)
        if pts.shape[0] < 10:
            print(f"Skipping {fname}: too few points")
            continue

        plt.figure(figsize=(4, 2))
        plt.plot(pts[:, 0], pts[:, 1], "k-", lw=1.5)
        plt.axis("equal")
        plt.xlabel("x")
        plt.ylabel("y")
        plt.title(fname.replace(".dat", ""))
        plt.grid(False)

        out_path = os.path.join(out_dir, fname.replace(".dat", ".png"))
        plt.savefig(out_path, dpi=dpi, bbox_inches="tight")
        plt.close()

    except Exception as e:
        print(f"Failed to process {fname}: {e}")

print(f"Saved {len(files)} airfoil images to '{out_dir}'")
