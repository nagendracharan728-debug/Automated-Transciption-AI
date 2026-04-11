import os
import json
import numpy as np
import matplotlib.pyplot as plt
import segeval
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# ---------------- CONFIG ----------------
TRANSCRIPT_DIR = "transcripts_cleaned"
GOLD_DIR = "gold_boundaries"
GRAPH_DIR = "similarity_graphs"
MODEL_NAME = "all-MiniLM-L6-v2"

os.makedirs(GRAPH_DIR, exist_ok=True)

print("Loading SBERT model...")
model = SentenceTransformer(MODEL_NAME)

results = {}

# ------------ HELPER FUNCTION ------------
def boundaries_to_masses(boundaries, total):
    masses = []
    prev = 0
    for b in sorted(boundaries):
        masses.append(b - prev)
        prev = b
    masses.append(total - prev)
    return tuple(masses)

# ------------ PROCESS EACH FILE ------------
for file in os.listdir(TRANSCRIPT_DIR):

    if not file.endswith(".json"):
        continue

    episode = file.replace(".json", "")
    print("\n==============================")
    print("Processing:", episode)

    # -------- Load Transcript --------
    transcript_path = os.path.join(TRANSCRIPT_DIR, file)

    try:
        with open(transcript_path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print("Error loading transcript:", e)
        continue

    if "segments" not in data:
        print("No 'segments' key found. Skipping.")
        continue

    segments = data["segments"]
    texts = [s["text"] for s in segments if "text" in s]

    print("Total segments:", len(texts))

    if len(texts) < 3:
        print("Too few sentences. Skipping.")
        continue

    # -------- Compute Embeddings --------
    embeddings = model.encode(texts)

    # -------- Compute Cosine Similarities --------
    sims = []
    for i in range(len(embeddings) - 1):
        sim = cosine_similarity(
            [embeddings[i]],
            [embeddings[i+1]]
        )[0][0]
        sims.append(sim)

    sims = np.array(sims)

    if len(sims) == 0:
        print("No similarity values computed.")
        continue

    # -------- Smooth --------
    window = min(5, len(sims))
    sims = np.convolve(sims, np.ones(window)/window, mode='same')

    # -------- Dynamic Threshold --------
    threshold = np.mean(sims) - np.std(sims) * 0.7

    predicted_boundaries = [
        i+1 for i, v in enumerate(sims) if v < threshold
    ]

    print("Predicted boundaries:", predicted_boundaries)

    # ==================================================
    # 🔵 ALWAYS PLOT (even if gold missing)
    # ==================================================
    plt.figure(figsize=(12, 6))
    plt.plot(sims)
    plt.axhline(y=threshold, linestyle='--')

    for b in predicted_boundaries:
        plt.axvline(x=b-1, linestyle='--')

    plt.title(f"SBERT Cosine Similarity - {episode}")
    plt.xlabel("Sentence Index")
    plt.ylabel("Cosine Similarity")
    plt.grid(True)

    save_path = os.path.join(GRAPH_DIR, episode + ".png")
    plt.savefig(save_path)
    plt.close()

    print("Graph saved at:", save_path)

    # ==================================================
    # 🔵 GOLD EVALUATION (Optional)
    # ==================================================
    gold_path = os.path.join(GOLD_DIR, file)

    if not os.path.exists(gold_path):
        print("Gold file not found. Skipping evaluation.")
        continue

    try:
        with open(gold_path, encoding="utf-8") as f:
            gold_data = json.load(f)
    except Exception as e:
        print("Error loading gold file:", e)
        continue

    # Robust gold extraction
    if isinstance(gold_data, dict) and "boundaries" in gold_data:
        gold_boundaries = gold_data["boundaries"]

    elif isinstance(gold_data, dict) and "gold_boundaries" in gold_data:
        gold_boundaries = gold_data["gold_boundaries"]

    elif isinstance(gold_data, list):
        gold_boundaries = gold_data

    else:
        print("Unknown gold format. Skipping evaluation.")
        continue

    print("Gold boundaries:", gold_boundaries)

    try:
        gold_m = boundaries_to_masses(gold_boundaries, len(texts))
        pred_m = boundaries_to_masses(predicted_boundaries, len(texts))

        pk = segeval.pk(gold_m, pred_m)
        wd = segeval.window_diff(gold_m, pred_m)

        gold_set = set(gold_boundaries)
        pred_set = set(predicted_boundaries)

        tp = len(gold_set & pred_set)

        precision = tp / len(pred_set) if pred_set else 0
        recall = tp / len(gold_set) if gold_set else 0

        results[episode] = {
            "Pk": round(pk, 4),
            "WindowDiff": round(wd, 4),
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "Predicted_Boundaries": predicted_boundaries
        }

        print("Evaluation done.")

    except Exception as e:
        print("SegEval error:", e)

# -------- Save Final Results --------
with open("sbert_evaluation_results.json", "w", encoding="utf-8") as f:
    json.dump(results, f, indent=2)

print("\nAll episodes processed successfully ✅")