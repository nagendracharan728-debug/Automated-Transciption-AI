import json
import os

INPUT_FOLDER = "segments_texttiling"        
OUTPUT_FOLDER = "boundaries_texttiling"

os.makedirs(OUTPUT_FOLDER, exist_ok=True)

for file in os.listdir(INPUT_FOLDER):
    if not file.endswith(".json"):
        continue

    path = os.path.join(INPUT_FOLDER, file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["segments"]

   
    boundaries = [seg["end_time"] for seg in segments[:-1]]

    out_file = file.replace(".json", "_boundaries.json")
    out_path = os.path.join(OUTPUT_FOLDER, out_file)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(boundaries, f, indent=2)

    print(f"✔ Saved boundaries for {file}")