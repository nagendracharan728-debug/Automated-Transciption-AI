import os
import json
import re

INPUT_DIR = "transcripts_cleaned"      # your Faster Whisper JSON folder
OUT_DIR = "sentences"
os.makedirs(OUT_DIR, exist_ok=True)

def split_into_sentences(text):
    # safe rule-based sentence split
    parts = re.split(r'(?<=[.!?])\s+', text.strip())
    return [p.strip() for p in parts if len(p.strip()) > 0]

for file in os.listdir(INPUT_DIR):

    if not file.endswith(".json"):
        continue

    path = os.path.join(INPUT_DIR, file)

    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    segments = data["segments"]   # your format confirmed

    sent_list = []
    sid = 1

    for seg in segments:

        text = seg["text"].strip()
        start = float(seg["start"])
        end = float(seg["end"])

        sentences = split_into_sentences(text)

        if len(sentences) == 0:
            continue

        duration = end - start
        chunk = duration / len(sentences)

        for i, s in enumerate(sentences):

            sent_list.append({
                "sentence_id": sid,
                "start_time": round(start + i*chunk, 2),
                "end_time": round(start + (i+1)*chunk, 2),
                "text": s
            })

            sid += 1

    output = {
        "episode_id": file.replace(".json",""),
        "duration": data.get("duration", None),
        "sentences": sent_list
    }

    out_path = os.path.join(OUT_DIR, file)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

print("✅ Sentence-level dataset created successfully")