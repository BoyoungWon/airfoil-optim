import os
import imageio.v2 as imageio

# ===================== CONFIG =====================

img_dir = "airfoil_images_1000"
out_video = "airfoil_fastforward_25fps.mp4"
fps = 10

# ===================== LOAD FILES =====================

files = sorted(f for f in os.listdir(img_dir) if f.endswith(".png"))
assert len(files) > 0, "No PNG images found"

# ===================== WRITE VIDEO =====================

with imageio.get_writer(
    out_video,
    fps=fps,
    codec="libx264",
    quality=8,          # good quality, small size
    pixelformat="yuv420p"
) as writer:
    for f in files:
        img = imageio.imread(os.path.join(img_dir, f))
        writer.append_data(img)

print(f"Saved video: {out_video} ({len(files)} frames @ {fps} fps)")
