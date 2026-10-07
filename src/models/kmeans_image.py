from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.cluster import KMeans


# Project paths
ROOT = Path(__file__).resolve().parents[2]
IMAGE = ROOT / "src" / "data" / "input_image.jpg"
OUT = ROOT / "reports" / "figures"

OUT.mkdir(parents=True, exist_ok=True)


# Load and resize image
img = Image.open(IMAGE).convert("RGB")
img = img.resize((300, 300))

# Convert image into RGB pixels
arr = np.array(img)
pixels = arr.reshape(-1, 3)


# Different K values
ks = [2, 4, 6, 8]

inertias = []
segmented_images = []


for k in ks:
    print(f"Running K-Means with K={k}...")

    model = KMeans(
        n_clusters=k,
        n_init=10,
        random_state=42
    )

    labels = model.fit_predict(pixels)

    # Replace each pixel with its cluster center color
    segmented = model.cluster_centers_[labels]
    segmented = segmented.reshape(arr.shape).astype(np.uint8)

    segmented_images.append(segmented)
    inertias.append(model.inertia_)

    # Save individual segmented image
    Image.fromarray(segmented).save(
        OUT / f"segmented_k{k}.png"
    )


# Comparison of all K values
fig, axes = plt.subplots(1, 5, figsize=(18, 4))

axes[0].imshow(arr)
axes[0].set_title("Original")
axes[0].axis("off")

for i, k in enumerate(ks):
    axes[i + 1].imshow(segmented_images[i])
    axes[i + 1].set_title(f"K = {k}")
    axes[i + 1].axis("off")

plt.tight_layout()
plt.savefig(
    OUT / "image_segmentation_comparison.png",
    dpi=150
)
plt.close()


# Elbow curve
plt.figure(figsize=(7, 5))
plt.plot(ks, inertias, marker="o")
plt.xlabel("Number of Clusters (K)")
plt.ylabel("Inertia")
plt.title("K-Means Elbow Curve")
plt.grid(True)
plt.savefig(
    OUT / "image_segmentation_elbow.png",
    dpi=150
)
plt.close()


print("\nImage segmentation completed successfully!")
print(f"Results saved in: {OUT}")