import os
import json
import matplotlib.pyplot as plt

GOLD_DIR = "gold_boundaries"
MINILM_DIR = "segments_minilm"
TEXTTILING_DIR = "segments_texttiling"
OUT_DIR = "results/plots"

os.makedirs(OUT_DIR, exist_ok=True)


def load_boundaries(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "true_boundaries" in data:
        return sorted([float(b) for b in data["true_boundaries"]])
    elif "boundaries" in data:
        return sorted([float(b) for b in data["boundaries"]])
    elif "segments" in data:
        return sorted([float(seg["end_time"]) for seg in data["segments"]])
    else:
        return []


for file in os.listdir(GOLD_DIR):

    if not file.endswith(".json"):
        continue

    episode = file.replace(".json", "")
    print("Visualizing:", episode)

    gold_path = os.path.join(GOLD_DIR, file)
    minilm_path = os.path.join(MINILM_DIR, file)
    tt_path = os.path.join(TEXTTILING_DIR, file)

    gold = load_boundaries(gold_path)
    minilm = load_boundaries(minilm_path) if os.path.exists(minilm_path) else []
    texttiling = load_boundaries(tt_path) if os.path.exists(tt_path) else []

   
    all_times = gold + minilm + texttiling
    if not all_times:
        continue

    max_time = max(all_times)

    plt.figure(figsize=(14, 4))

   
    for t in gold:
        plt.axvline(t, ymin=0.6, ymax=0.9, linestyle='-', label="Gold" if t == gold[0] else "")

    
    for t in minilm:
        plt.axvline(t, ymin=0.3, ymax=0.6, linestyle='--', label="MiniLM" if t == minilm[0] else "")

    
    for t in texttiling:
        plt.axvline(t, ymin=0.0, ymax=0.3, linestyle=':', label="TextTiling" if t == texttiling[0] else "")

    plt.ylim(0, 1)
    plt.xlim(0, max_time)

    plt.yticks([])
    plt.xlabel("Time (seconds)")
    plt.title(f"Topic Boundary Comparison - {episode}")

    plt.legend(loc="upper right")

    plt.tight_layout()
    plt.savefig(os.path.join(OUT_DIR, f"{episode}_boundaries.png"))
    plt.close()

print("\nAll boundary plots saved in results/plots/")