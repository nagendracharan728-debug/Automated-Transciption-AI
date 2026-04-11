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


similarity_stats = []

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
        sims.append(cosine_similarity([emb[i]],[emb[i+1]])[0][0])
    sims = np.array(sims)
    sims_smooth = smooth(sims, 5)

    # save similarity stats
    sim_mean = np.mean(sims_smooth)
    sim_std = np.std(sims_smooth)
    similarity_stats.append({
        "episode": file.replace(".json",""),
        "mean_similarity": float(sim_mean),
        "std_similarity": float(sim_std),
        "min_similarity": float(np.min(sims_smooth)),
        "max_similarity": float(np.max(sims_smooth))
    })

    # boundary detection
    threshold = sim_mean - sim_std*0.7
    boundaries = [times[i+1] for i,v in enumerate(sims_smooth) if v < threshold]

    out = {
        "episode_id": file.replace(".json",""),
        "boundaries": boundaries
    }

    with open(os.path.join(OUT_DIR,file),"w") as f:
        json.dump(out,f, indent=2)

# Save all similarity stats to JSON
with open(os.path.join(OUT_DIR,"similarity_stats.json"),"w") as f:
    json.dump(similarity_stats,f,indent=2)

print("Saved similarity stats for all episodes.")