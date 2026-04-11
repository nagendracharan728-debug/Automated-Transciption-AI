import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

TRANSCRIPT_DIR = "transcripts_cleaned"
OUT_DIR = "segments_sbert"
os.makedirs(OUT_DIR, exist_ok=True)

model = SentenceTransformer("all-MiniLM-L6-v2")

def smooth(x, w=3):
    return np.convolve(x, np.ones(w)/w, mode='same')

for file in os.listdir(TRANSCRIPT_DIR):
    if not file.endswith(".json"):
        continue

    with open(os.path.join(TRANSCRIPT_DIR,file)) as f:
        data = json.load(f)

    segments = data["segments"]
    texts = [s["text"] for s in segments]
    times = [s["start"] for s in segments]

    emb = model.encode(texts)

    sims = []
    for i in range(len(emb)-1):
        sims.append(cosine_similarity([emb[i]], [emb[i+1]])[0][0])

    sims = np.array(sims)
    sims = smooth(sims,5)

    # boundary = low similarity points
    threshold = np.mean(sims) - np.std(sims)*0.7
    boundary_times = []
    for i,v in enumerate(sims):
        if v < threshold:
            boundary_times.append(times[i+1])

    # --- create segments in same format as MiniLM ---
    all_segments = []
    start = 0.0
    segment_id = 1
    for b in boundary_times + [segments[-1]["start"] + 1]:  # add last timestamp
        end = float(b)
        text_slice = " ".join([s["text"] for s in segments if s["start"] >= start and s["start"] < end])
        all_segments.append({
            "segment_id": segment_id,
            "start_time": float(start),
            "end_time": float(end),
            "text": text_slice,
            "embedding_model": "SBERT-MiniLM-L6-v2"
        })
        segment_id += 1
        start = end

    out = {
        "episode_id": file.replace(".json",""),
        "segments": all_segments
    }

    with open(os.path.join(OUT_DIR,file), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)

    print("Segmented:", file)