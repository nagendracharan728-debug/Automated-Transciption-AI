import os
import json
import pandas as pd

METRIC_DIR = "results/metrics"
OUT = "results/tables/final_comparison.csv"

rows = []

for file in os.listdir(METRIC_DIR):

    if not file.endswith(".json"):
        continue

    
    if file == "all_results_summary.json":
        continue

    episode = file.replace(".json", "")

    with open(os.path.join(METRIC_DIR, file)) as f:
        data = json.load(f)

    row = {
        "Episode": episode,

        
        "MiniLM_Pk": data.get("MiniLM", {}).get("Pk"),
        "MiniLM_WD": data.get("MiniLM", {}).get("WindowDiff"),
        "MiniLM_Precision": data.get("MiniLM", {}).get("Precision"),
        "MiniLM_Recall": data.get("MiniLM", {}).get("Recall"),

        "TextTiling_Pk": data.get("TextTiling", {}).get("Pk"),
        "TextTiling_WD": data.get("TextTiling", {}).get("WindowDiff"),
        "TextTiling_Precision": data.get("TextTiling", {}).get("Precision"),
        "TextTiling_Recall": data.get("TextTiling", {}).get("Recall"),
    }

    rows.append(row)

df = pd.DataFrame(rows)

os.makedirs("results/tables", exist_ok=True)
df.to_csv(OUT, index=False)

print("Table saved:", OUT)
print(df)