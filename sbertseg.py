import os
import json
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from keybert import KeyBERT
from transformers import pipeline

# ===============================
# CONFIG
# ===============================

TRANSCRIPT_DIR = "transcripts_cleaned"
OUTPUT_DIR = "FINAL_BLOCK5_RESULTS"

BLOCK_SIZES = [5]
K_VALUES = [0.8, 1.0, 1.2]

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ===============================
# LOAD MODELS
# ===============================

print("Loading MiniLM...")
embedder = SentenceTransformer("all-MiniLM-L6-v2")

print("Loading KeyBERT...")
kw_model = KeyBERT()

print("Loading BART summarizer...")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

print("Loading Sentiment Model...")
sentiment_model = pipeline("sentiment-analysis")

print("Models Loaded ✅\n")

# ===============================
# HELPER FUNCTIONS
# ===============================

def load_transcript(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)["segments"]

def get_sentences_times(segments):
    sentences = []
    times = []
    for seg in segments:
        if seg["text"].strip():
            sentences.append(seg["text"].strip())
            times.append((seg["start"], seg["end"]))
    return sentences, times

def create_blocks(sentences, block_size):
    blocks = []
    for i in range(0, len(sentences), block_size):
        blocks.append(" ".join(sentences[i:i+block_size]))
    return blocks

def compute_boundaries(blocks, times, block_size, k):
    embeddings = embedder.encode(blocks)
    sims = []

    for i in range(len(embeddings)-1):
        sim = cosine_similarity(
            [embeddings[i]], [embeddings[i+1]]
        )[0][0]
        sims.append(sim)

    mean = np.mean(sims)
    std = np.std(sims)
    threshold = mean - (k * std)

    boundaries = []

    for i, sim in enumerate(sims):
        if sim < threshold:
            idx = (i+1)*block_size
            if idx < len(times):
                boundaries.append(times[idx][0])

    return boundaries

def create_segments(sentences, times, boundaries):

    segments = []
    current_text = []
    start_time = times[0][0]
    segment_id = 1
    b_idx = 0

    for i, sentence in enumerate(sentences):

        current_text.append(sentence)

        if b_idx < len(boundaries) and times[i][0] >= boundaries[b_idx]:

            end_time = times[i][1]

            segments.append({
                "segment_id": segment_id,
                "start_time": round(start_time,2),
                "end_time": round(end_time,2),
                "text": " ".join(current_text)
            })

            segment_id += 1
            current_text = []
            start_time = times[i][1]
            b_idx += 1

    if current_text:
        segments.append({
            "segment_id": segment_id,
            "start_time": round(start_time,2),
            "end_time": round(times[-1][1],2),
            "text": " ".join(current_text)
        })

    return segments

def extract_keywords(text):
    kws = kw_model.extract_keywords(text, top_n=5)
    return [kw[0] for kw in kws]

# ===============================
# SENTIMENT FUNCTION
# ===============================

def get_sentiment_score(text):

    result = sentiment_model(text[:512])[0]

    label = result["label"]
    score = result["score"]

    if label == "NEGATIVE":
        score = -score

    return round(score,3)

from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("facebook/bart-large-cnn")

def summarize_text(text):

    inputs = tokenizer(
        text,
        max_length=1024,
        truncation=True,
        return_tensors="pt"
    )

    input_length = inputs["input_ids"].shape[1]

    max_len = max(15, int(input_length * 0.4))
    min_len = max(8, int(input_length * 0.2))

    if max_len >= input_length:
        max_len = max(10, input_length - 5)

    summary_ids = summarizer.model.generate(
        inputs["input_ids"],
        max_length=max_len,
        min_length=min_len,
        do_sample=False
    )

    summary = tokenizer.decode(
        summary_ids[0],
        skip_special_tokens=True
    )

    return summary

# ===============================
# PROCESS ALL PODCASTS
# ===============================

for file in os.listdir(TRANSCRIPT_DIR):

    if not file.endswith(".json"):
        continue

    print(f"\nProcessing {file}")

    transcript_path = os.path.join(TRANSCRIPT_DIR, file)
    raw_segments = load_transcript(transcript_path)

    sentences, times = get_sentences_times(raw_segments)

    episode_id = file.replace(".json","")

    episode_results = []

    for block in BLOCK_SIZES:
        blocks = create_blocks(sentences, block)

        for k in K_VALUES:

            print(f"  Block={block}, k={k}")

            boundaries = compute_boundaries(blocks, times, block, k)

            structured_segments = create_segments(
                sentences, times, boundaries
            )

            for seg in structured_segments:

                seg["keywords"] = extract_keywords(seg["text"])
                seg["summary"] = summarize_text(seg["text"])
                seg["sentiment_score"] = get_sentiment_score(seg["text"])

            episode_results.append({
                "block_size": block,
                "k_value": k,
                "number_of_segments": len(structured_segments),
                "segments": structured_segments
            })

    with open(
        os.path.join(OUTPUT_DIR, f"{episode_id}_final.json"),
        "w",
        encoding="utf-8"
    ) as f:

        json.dump({
            "episode_id": episode_id,
            "all_configurations": episode_results
        }, f, indent=2)

print("\n✅ ALL PODCASTS PROCESSED")