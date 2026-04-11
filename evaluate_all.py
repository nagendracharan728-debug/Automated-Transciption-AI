import os
import json
import numpy as np
import segeval

GOLD_DIR = "gold_boundaries"
MINILM_DIR = "segments_minilm"
TEXTTILING_DIR = "segments_texttiling"

OUT_DIR = "results/metrics"
os.makedirs(OUT_DIR, exist_ok=True)




def load_gold(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "true_boundaries" in data:
        return sorted([float(b) for b in data["true_boundaries"]])
    elif "segments" in data:
        return sorted([float(seg["end_time"]) for seg in data["segments"]])
    else:
        raise ValueError("No boundaries found")


def load_predicted(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if "boundaries" in data:
        return sorted([float(b) for b in data["boundaries"]])
    elif "segments" in data:
        return sorted([float(seg["end_time"]) for seg in data["segments"]])
    else:
        raise ValueError("No boundaries found")




def boundaries_to_masses(boundaries, duration):
    """
    Convert boundaries to integer segment lengths
    while ensuring total sum equals duration exactly.
    """

    boundaries = [0.0] + boundaries + [duration]
    float_lengths = []

    for i in range(len(boundaries) - 1):
        length = boundaries[i+1] - boundaries[i]
        if length > 0:
            float_lengths.append(length)


    int_lengths = [int(round(x)) for x in float_lengths]

   
    diff = int(round(duration)) - sum(int_lengths)

    
    if len(int_lengths) > 0:
        int_lengths[-1] += diff

    return int_lengths



def compute_metrics(gold, pred):

    if len(gold) == 0 or len(pred) == 0:
        return {
            "Pk": None,
            "WindowDiff": None,
            "Precision": None,
            "Recall": None
        }

    
    duration = max(max(gold), max(pred))

    gold_m = boundaries_to_masses(gold, duration)
    pred_m = boundaries_to_masses(pred, duration)

    try:
        pk = float(segeval.pk(gold_m, pred_m))
        wd = float(segeval.window_diff(gold_m, pred_m))
    except Exception as e:
        print("segeval error:", e)
        pk = None
        wd = None

    gold_set = set(np.round(gold, 1))
    pred_set = set(np.round(pred, 1))

    tp = len(gold_set & pred_set)

    precision = tp / len(pred_set) if pred_set else 0
    recall = tp / len(gold_set) if gold_set else 0

    return {
        "Pk": pk,
        "WindowDiff": wd,
        "Precision": float(precision),
        "Recall": float(recall)
    }




for file in os.listdir(GOLD_DIR):

    if not file.endswith(".json"):
        continue

    episode = file.replace(".json", "")
    print(f"\nEvaluating Episode: {episode}")

    gold_path = os.path.join(GOLD_DIR, file)
    minilm_path = os.path.join(MINILM_DIR, file)
    tt_path = os.path.join(TEXTTILING_DIR, file)

    try:
        gold = load_gold(gold_path)
    except Exception as e:
        print("Gold load error:", e)
        continue

    results = {}

    if os.path.exists(minilm_path):
        pred = load_predicted(minilm_path)
        results["MiniLM"] = compute_metrics(gold, pred)

    if os.path.exists(tt_path):
        pred = load_predicted(tt_path)
        results["TextTiling"] = compute_metrics(gold, pred)

    if not results:
        print("No predictions found. Skipping.")
        continue

    out_path = os.path.join(OUT_DIR, file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    print("Done:", episode)

print("\nAll evaluations completed successfully.")