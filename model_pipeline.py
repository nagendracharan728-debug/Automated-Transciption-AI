import json
import os
from speech_to_text import audio_to_transcript
from transformers import pipeline
from keybert import KeyBERT

# Load NLP models
summarizer = pipeline("summarization")
sentiment_model = pipeline("sentiment-analysis")
kw_model = KeyBERT()

def process_podcast(filepath):

    ext = filepath.split(".")[-1].lower()

    # AUDIO FILE
    if ext in ["mp3", "wav", "m4a"]:
        transcript = audio_to_transcript(filepath)

    # JSON FILE
    elif ext == "json":
        with open(filepath, "r", encoding="utf-8") as f:
            transcript = json.load(f)

    else:
        raise ValueError("Unsupported file type")

    # Convert transcript to text if it's dict
    if isinstance(transcript, dict):
        text = transcript.get("text", "")
    else:
        text = transcript

    # -------- SEGMENTATION --------
    segments = text.split(". ")

    results = []

    for seg in segments:

        if len(seg.strip()) < 10:
            continue

        # -------- KEYWORDS --------
        keywords = kw_model.extract_keywords(seg, top_n=3)
        keywords = [k[0] for k in keywords]

        # -------- SUMMARY --------
        try:
            summary = summarizer(seg, max_length=30, min_length=5, do_sample=False)[0]["summary_text"]
        except:
            summary = seg

        # -------- SENTIMENT --------
        sentiment = sentiment_model(seg)[0]

        results.append({
            "segment": seg,
            "keywords": keywords,
            "summary": summary,
            "sentiment": sentiment["label"],
            "sentiment_score": float(sentiment["score"])
        })

    os.makedirs("outputs", exist_ok=True)

    output_path = os.path.join("outputs", "result.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4)

    return output_path