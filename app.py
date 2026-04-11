from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for
import os, gc, csv, base64
from io import BytesIO
from faster_whisper import WhisperModel
from sentence_transformers import SentenceTransformer, util
from transformers import pipeline
from textblob import TextBlob
import yake
from wordcloud import WordCloud
import re
from flask_socketio import SocketIO
app = Flask(__name__)

socketio = SocketIO(app, cors_allowed_origins="*", async_mode="threading")
# AI MODELS

stt_model = WhisperModel("tiny", device="cpu", compute_type="int8", cpu_threads=4)
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
kw_extractor = yake.KeywordExtractor(lan="en", n=1, top=5)
summarizer = pipeline("summarization", model="sshleifer/distilbart-cnn-6-6", device=-1)

# GLOBAL STORAGE

transcript_texts = []
transcript_embeddings = None
segments_global = []
graph_data_global = {}

# UPLOAD FOLDER

UPLOAD_FOLDER = "static/audio"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def stream_transcription(filepath):
    print("🚀 Streaming started:", filepath)
    segments, _ = stt_model.transcribe(filepath, beam_size=1, vad_filter=True)

    for s in segments:
        text = re.sub(r'\s+', ' ', s.text)
        text = re.sub(r'[^A-Za-z0-9.,?! ]+', '', text)
        text = text.strip()

        print("📤 Sending:", text)

        socketio.emit("new_caption", {
            "start": round(s.start, 2),
            "end": round(s.end, 2),
            "text": text
        }, broadcast=True)

        socketio.sleep(0.1)
@socketio.on("start_transcription")
def handle_transcription(data):
    filename = data.get("filename")

    print("🔥 START EVENT RECEIVED:", filename)

    if not filename:
        print("❌ No filename received")
        return

    filepath = os.path.join(UPLOAD_FOLDER, filename)

    if not os.path.exists(filepath):
        print("❌ File not found:", filepath)
        return

    print("✅ File found, starting transcription...")

    socketio.start_background_task(stream_transcription, filepath)
#  SPEECH TO TEXT

def transcribe_audio(filepath):
    segments, _ = stt_model.transcribe(filepath, beam_size=1, vad_filter=True)

    cleaned_segments = []
    for s in segments:
        text = re.sub(r'\s+', ' ', s.text)
        text = re.sub(r'[^A-Za-z0-9.,?! ]+', '', text)
        text = text.strip()
        cleaned_segments.append({"start": s.start, "end": s.end, "text": text})

    return cleaned_segments

#  BLOCK SEGMENTATION

def block_segmentation(segments, block_size=6):
    blocks = []
    for i in range(0, len(segments), block_size):
        chunk = segments[i:i + block_size]
        text = " ".join([s["text"] for s in chunk])
        blocks.append({
            "text": text,
            "start": chunk[0]["start"],
            "end": chunk[-1]["end"]
        })
    return blocks

#  TOPIC SEGMENTATION

def topic_segmentation(blocks, similarity_threshold=0.75, min_duration=40):
    texts = [b["text"] for b in blocks]

    embeddings = embed_model.encode(texts, convert_to_tensor=False, batch_size=4)

    merged = [blocks[0]]

    for i in range(1, len(blocks)):
        sim = util.cos_sim(embeddings[i], embeddings[i - 1]).item()
        duration = merged[-1]["end"] - merged[-1]["start"]

        if sim >= similarity_threshold or duration < min_duration:
            merged[-1]["text"] += " " + blocks[i]["text"]
            merged[-1]["end"] = blocks[i]["end"]
        else:
            merged.append(blocks[i])

    return merged

#  SUMMARIZATION

def generate_summary(text):
    try:
        words = text.split()

        if len(words) < 40:
            return text

        text = " ".join(words[:120])

        summary_text = summarizer(
            text,
            max_length=80,
            min_length=30,
            do_sample=False,
            num_beams=2
        )[0]['summary_text']

        sentences = re.split(r'(?<=[.?!]) +', summary_text)
        bullets = "\n".join(["• " + s.strip() for s in sentences if s.strip()])

        return bullets

    except Exception as e:
        print("Summarization Error:", e)
        return " ".join(text.split()[:60])


#  ANALYZE SEGMENTS

def analyze_segments(blocks):
    segments, sentiments = [], []

    for idx, block in enumerate(blocks, start=1):
        text = block["text"]
        summary = generate_summary(text)

        keywords = [k[0] for k in kw_extractor.extract_keywords(text)]
        topics = keywords[:2]

        polarity = TextBlob(text).sentiment.polarity
        sentiment_label = "positive" if polarity > 0.1 else "negative" if polarity < -0.1 else "neutral"

        sentiments.append(sentiment_label)

        segments.append({
            "segment_id": idx,
            "start_time": round(block["start"], 2),
            "end_time": round(block["end"], 2),
            "text": text,
            "summary": summary,
            "keywords": keywords,
            "topics": topics,
            "sentiment_score": polarity,
            "sentiment_label": sentiment_label
        })

    return segments, sentiments


#  SENTIMENT STATS

def sentiment_statistics(sentiments):
    total = len(sentiments)
    return {
        "positive": round(sentiments.count("positive") / total * 100) if total else 0,
        "neutral": round(sentiments.count("neutral") / total * 100) if total else 0,
        "negative": round(sentiments.count("negative") / total * 100) if total else 0
    }


#  WORDCLOUD

def generate_wordcloud(segments):
    all_text = " ".join([s["text"] for s in segments])

    wc = WordCloud(width=800, height=400, background_color="black", colormap="Set2").generate(all_text)

    buffer = BytesIO()
    wc.to_image().save(buffer, format="PNG")

    return base64.b64encode(buffer.getvalue()).decode("utf-8")


#  EXPORT CSV

def export_csv(segments):
    import io
    buffer = io.StringIO()
    writer = csv.writer(buffer)

    writer.writerow(["Segment ID", "Start Time", "End Time", "Text", "Summary", "Keywords", "Sentiment"])

    for s in segments:
        writer.writerow([
            s["segment_id"],
            s["start_time"],
            s["end_time"],
            s["text"],
            s["summary"],
            ", ".join(s["keywords"]),
            s["sentiment_label"]
        ])

    return BytesIO(buffer.getvalue().encode())


# ROUTES

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/upload")
def upload():
    return render_template("upload.html")

@app.route("/dashboard")
def dashboard():
    global graph_data_global

    return render_template(
        "dashboard.html",
        segments=segments_global,
        audio_file=graph_data_global.get("audio_file", None),
        sentiment=graph_data_global.get("sentiment", {"positive": 0, "neutral": 0, "negative": 0}),
        segment_labels=graph_data_global.get("segment_labels", []),
        segment_durations=graph_data_global.get("segment_durations", []),
        topic_boundaries=graph_data_global.get("topic_boundaries", []),
        sentiment_times=graph_data_global.get("sentiment_times", []),
        sentiment_scores=graph_data_global.get("sentiment_scores", []),
        keyword_words=graph_data_global.get("keyword_words", []),
        keyword_counts=graph_data_global.get("keyword_counts", []),
        wordcloud_image=graph_data_global.get("wordcloud_image", None),
        segment_colors=[]
    )

@app.route("/process", methods=["POST"])
def process():
    file = request.files.get("podcast")

    if not file or file.filename == "":
        return "No file uploaded.", 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    segments_raw = transcribe_audio(filepath)
    blocks = block_segmentation(segments_raw)
    topic_blocks = topic_segmentation(blocks)
    segments, sentiments = analyze_segments(topic_blocks)

    global transcript_texts, transcript_embeddings, segments_global, graph_data_global

    transcript_texts = [s["text"] for s in segments][:40]
    transcript_embeddings = embed_model.encode(transcript_texts, convert_to_tensor=False, batch_size=4)

    segments_global = segments

    del segments_raw
    gc.collect()

    sentiment_counts = sentiment_statistics(sentiments)

    segment_labels = [s["segment_id"] for s in segments]
    segment_durations = [s["end_time"] - s["start_time"] for s in segments]
    topic_boundaries = [s["end_time"] for s in segments]
    sentiment_times = [s["start_time"] for s in segments]
    sentiment_scores = [s["sentiment_score"] for s in segments]

    all_keywords = [k for seg in segments for k in seg["keywords"]]
    keyword_words = list(set(all_keywords))
    keyword_counts = [all_keywords.count(k) for k in keyword_words]

    wordcloud_image = generate_wordcloud(segments)

    graph_data_global = {
        "segment_labels": segment_labels,
        "segment_durations": segment_durations,
        "topic_boundaries": topic_boundaries,
        "sentiment_times": sentiment_times,
        "sentiment_scores": sentiment_scores,
        "keyword_words": keyword_words,
        "keyword_counts": keyword_counts,
        "wordcloud_image": wordcloud_image,
        "sentiment": sentiment_counts,
        "audio_file": file.filename
    }

    return redirect(url_for("dashboard"))
@app.route("/upload_audio", methods=["POST"])
def upload_audio():
    file = request.files.get("podcast")

    if not file:
        return jsonify({"error": "No file"}), 400

    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    filename = file.filename
    print("✅ Uploaded file:", filename)
    print("📁 Saved at:", filepath)

    # 🔥 ADD THIS PART (MAIN FIX)
    print("⏳ Generating captions...")
    segments = transcribe_audio(filepath)
    print("✅ Captions generated")

    return jsonify({
        "filename": filename,
        "captions": segments   # ✅ SEND CAPTIONS TO FRONTEND
    })

# OTHER ROUTES

@app.route("/export_csv")
def download_csv():
    if not segments_global:
        return "No segments to export", 400

    return send_file(export_csv(segments_global),
                     as_attachment=True,
                     download_name="podcast_segments.csv",
                     mimetype="text/csv")

@app.route("/realtime")
def realtime():
    return render_template("live_transcribe.html")

@app.route("/chatbot")
def chatbot_page():
    return render_template("chatbot.html")

@app.route("/ask_project_ai", methods=["POST"])
def ask_project_ai():
    data = request.get_json()
    question = data.get("question", "")

    global transcript_texts, transcript_embeddings

    if transcript_embeddings is None:
        return jsonify({"answer": "Please upload a podcast first."})

    question_embedding = embed_model.encode(question)
    scores = util.cos_sim(question_embedding, transcript_embeddings)[0]

    best_match = int(scores.argmax())
    answer = transcript_texts[best_match]

    return jsonify({"answer": answer})


if __name__ == "__main__":
    socketio.run(app, debug=True, use_reloader=False)