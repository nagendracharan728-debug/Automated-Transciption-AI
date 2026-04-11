import os
import json
from nltk.sentiment import SentimentIntensityAnalyzer
import nltk

# -----------------------------
# Correct Input Paths
# -----------------------------
SEGMENT_DIR = "final_output/segmentation"
KEYWORD_DIR = "final_output/keywords"
SUMMARY_DIR = "final_output/summaries"

OUTPUT_DIR = "final_schema_output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

nltk.download("vader_lexicon")
sia = SentimentIntensityAnalyzer()

print("VADER Loaded ✅")


# -----------------------------
# Sentiment Function
# -----------------------------
def compute_sentiment(text):
    return round(sia.polarity_scores(text)["compound"], 3)


# -----------------------------
# Timestamp Validation
# -----------------------------
def validate_timestamps(segments, duration):

    for i, seg in enumerate(segments):

        if seg["start_time"] >= seg["end_time"]:
            raise ValueError(f"Invalid timestamp in segment {seg['segment_id']}")

        if i > 0 and seg["start_time"] < segments[i-1]["end_time"]:
            raise ValueError("Segment overlap detected")

        if seg["end_time"] > duration:
            raise ValueError("Segment exceeds episode duration")

    return True


# -----------------------------
# Process Files
# -----------------------------
for file in os.listdir(SEGMENT_DIR):

    if not file.endswith(".json"):
        continue

    print(f"Processing {file}")

    # -----------------------------
    # Load Segmentation File
    # -----------------------------
    with open(os.path.join(SEGMENT_DIR, file), "r", encoding="utf-8") as f:
        seg_data = json.load(f)

    # Handle both possible structures
    if isinstance(seg_data, list):
        segments = seg_data
        episode_id = file.replace(".json", "")
    else:
        segments = seg_data.get("segments", [])
        episode_id = seg_data.get("episode_id", file.replace(".json", ""))

    if not segments:
        print("No segments found, skipping...")
        continue

    # Calculate duration automatically
    duration = segments[-1]["end_time"]

    # -----------------------------
    # Load Keywords (if exists)
    # -----------------------------
    keyword_path = os.path.join(KEYWORD_DIR, file)
    if os.path.exists(keyword_path):
        with open(keyword_path, "r", encoding="utf-8") as f:
            keyword_data = json.load(f)

        if isinstance(keyword_data, list):
            keyword_segments = keyword_data
        else:
            keyword_segments = keyword_data.get("segments", [])
    else:
        keyword_segments = segments

    # -----------------------------
    # Load Summaries (if exists)
    # -----------------------------
    summary_path = os.path.join(SUMMARY_DIR, file)
    if os.path.exists(summary_path):
        with open(summary_path, "r", encoding="utf-8") as f:
            summary_data = json.load(f)

        if isinstance(summary_data, list):
            summary_segments = summary_data
        else:
            summary_segments = summary_data.get("segments", [])
    else:
        summary_segments = segments

    # -----------------------------
    # Build Final Segments
    # -----------------------------
    final_segments = []

    for i, seg in enumerate(segments):

        sentiment_score = compute_sentiment(seg["text"])

        keywords = []
        if i < len(keyword_segments):
            keywords = keyword_segments[i].get("keywords", [])

        summary = ""
        if i < len(summary_segments):
            summary = summary_segments[i].get("summary", "")

        final_segments.append({
            "segment_id": seg["segment_id"],
            "start_time": seg["start_time"],
            "end_time": seg["end_time"],
            "text": seg["text"],
            "keywords": keywords,
            "summary": summary,
            "sentiment_score": sentiment_score
        })

    # Validate timestamps
    validate_timestamps(final_segments, duration)

    # -----------------------------
    # Final Output Structure
    # -----------------------------
    final_output = {
        "episode_id": episode_id,
        "duration": duration,
        "segments": final_segments
    }

    out_path = os.path.join(OUTPUT_DIR, file)

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(final_output, f, indent=2)

    print(f"Saved → {out_path}")

print("FINAL SCHEMA BUILD COMPLETE ✅")