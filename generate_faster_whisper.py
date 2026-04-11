from faster_whisper import WhisperModel
import os
import json

AUDIO_DIR = "cleaned_audio"
OUT_DIR = "transcripts_cleaned"

MODEL_SIZE = "base"      
DEVICE = "cpu"
COMPUTE_TYPE = "int8"

os.makedirs(OUT_DIR, exist_ok=True)

print(f"Loading Faster-Whisper model ({MODEL_SIZE})...")
model = WhisperModel(MODEL_SIZE, device=DEVICE, compute_type=COMPUTE_TYPE)
print("Model loaded. Starting transcription...\n")

files = sorted(os.listdir(AUDIO_DIR))

processed = 0
skipped = 0

for file in files:

    if not file.lower().endswith(".wav"):
        continue

    if "temp" in file.lower():
        continue

    input_path = os.path.join(AUDIO_DIR, file)
    base_name = os.path.splitext(file)[0]
    output_path = os.path.join(OUT_DIR, base_name + ".json")

    
    if os.path.exists(output_path):
        print(f"⏭ Skipping existing transcript: {file}")
        skipped += 1
        continue

    print(f"🎙 Transcribing with BASE model: {file}")

    try:
        segments, info = model.transcribe(
            input_path,
            beam_size=5,
            vad_filter=True
        )

        transcript_data = {
            "file": file,
            "model_used": MODEL_SIZE,   
            "language": info.language,
            "duration": info.duration,
            "segments": []
        }

        for segment in segments:
            transcript_data["segments"].append({
                "start": float(segment.start),
                "end": float(segment.end),
                "text": segment.text.strip()
            })

        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(transcript_data, f, indent=2, ensure_ascii=False)

        print(f" Saved → {output_path}\n")
        processed += 1

    except Exception as e:
        print(f" Error processing {file}: {e}\n")

print("\n==============================")
print(f" Newly processed : {processed}")
print(f"⏭ Already skipped : {skipped}")
print(" Done — medium outputs untouched.")
