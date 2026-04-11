import whisper
import json

model = whisper.load_model("base")

def audio_to_transcript(audio_path):

    result = model.transcribe(audio_path)

    transcript = {
        "episode_id": "uploaded_audio",
        "duration": result["segments"][-1]["end"],
        "segments": []
    }

    for seg in result["segments"]:

        transcript["segments"].append({
            "start_time": seg["start"],
            "end_time": seg["end"],
            "text": seg["text"]
        })

    return transcript