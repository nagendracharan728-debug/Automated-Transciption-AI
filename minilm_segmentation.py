import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

INPUT_DIR = "sentences"
OUT_DIR = "segments_minilm"
SIM_DIR = "similarity_values"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(SIM_DIR, exist_ok=True)

print("Loading MiniLM model...")
model = SentenceTransformer("all-MiniLM-L6-v2")

BLOCK_SIZE = 5

for file in os.listdir(INPUT_DIR):

    if not file.endswith(".json"):
        continue

    print("Processing:", file)

    with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = data["sentences"]

    # ---- STEP 1: CREATE BLOCKS ----
    blocks = []

    for i in range(0, len(sentences), BLOCK_SIZE):
        chunk = sentences[i:i+BLOCK_SIZE]

        blocks.append({
            "start_time": chunk[0]["start_time"],
            "end_time": chunk[-1]["end_time"],
            "text": " ".join(x["text"] for x in chunk)
        })

    texts = [b["text"] for b in blocks]

    # ---- STEP 2: EMBEDDINGS ----
    embeddings = model.encode(texts)

    # ---- STEP 3: SIMILARITY ----
    sims = []

    for i in range(len(embeddings)-1):
        sim = cosine_similarity(
            [embeddings[i]],
            [embeddings[i+1]]
        )[0][0]
        sims.append(sim)

    # save raw similarity values
    np.save(os.path.join(SIM_DIR, file.replace(".json",".npy")), sims)

    # ---- STEP 4: BOUNDARY DETECTION ----
    mean = np.mean(sims)
    std = np.std(sims)

    threshold = mean - std

    boundaries = []

    for i, s in enumerate(sims):
        if s < threshold:
            boundaries.append(i+1)

    # ---- STEP 5: CREATE SEGMENTS ----
    segs = []
    start_block = 0
    seg_id = 1

    for b in boundaries + [len(blocks)]:

        seg_blocks = blocks[start_block:b]

        segs.append({
            "segment_id": seg_id,
            "start_time": seg_blocks[0]["start_time"],
            "end_time": seg_blocks[-1]["end_time"],
            "text": " ".join(x["text"] for x in seg_blocks),
            "embedding_model": "MiniLM-L6-v2"
        })

        start_block = b
        seg_id += 1

    output = {
        "episode_id": data["episode_id"],
        "segments": segs
    }

    with open(os.path.join(OUT_DIR, file), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

print(" MiniLM segmentation completed")