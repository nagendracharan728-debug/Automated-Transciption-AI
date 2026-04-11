import os
import numpy as np
import matplotlib.pyplot as plt

SIM_DIR = "similarity_values"
OUT_DIR = "similarity_plots"

os.makedirs(OUT_DIR, exist_ok=True)

for file in os.listdir(SIM_DIR):

    if not file.endswith(".npy"):
        continue

    sims = np.load(os.path.join(SIM_DIR, file))

    name = file.replace(".npy","")

    plt.figure(figsize=(10,4))
    plt.plot(sims)
    plt.title(f"Similarity Curve - {name}")
    plt.xlabel("Block Index")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)

    save_path = os.path.join(OUT_DIR, f"{name}.png")
    plt.savefig(save_path)
    plt.close()

    print("Saved:", save_path)

print("\n✅ All similarity graphs saved")