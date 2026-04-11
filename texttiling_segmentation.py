import os
import json
from nltk.tokenize import TextTilingTokenizer

INPUT_DIR = "sentences"
OUT_DIR = "segments_texttiling"
os.makedirs(OUT_DIR, exist_ok=True)

tokenizer = TextTilingTokenizer()

for file in os.listdir(INPUT_DIR):

    if not file.endswith(".json"):
        continue

    with open(os.path.join(INPUT_DIR, file), "r", encoding="utf-8") as f:
        data = json.load(f)

    sentences = data["sentences"]

    # ✅ INSERT PARAGRAPH BREAKS every 8 sentences
    text_blocks = []
    block = []

    for i, s in enumerate(sentences):
        block.append(s["text"])

        if (i+1) % 8 == 0:
            text_blocks.append(" ".join(block))
            block = []

    if block:
        text_blocks.append(" ".join(block))

    full_text = "\n\n".join(text_blocks)

    # ---- Run TextTiling safely ----
    try:
        segments = tokenizer.tokenize(full_text)
    except:
        print(f"⚠ TextTiling failed on {file}, using fallback (single segment)")
        segments = [full_text]

    # ---- Convert to timestamps ----
    seg_list = []
    pointer = 0
    seg_id = 1

    for seg in segments:

        words = seg.split()
        approx_sentences = max(1, len(words) // 15)

        start_time = sentences[pointer]["start_time"]

        pointer_end = min(pointer + approx_sentences, len(sentences)-1)
        end_time = sentences[pointer_end]["end_time"]

        seg_list.append({
            "segment_id": seg_id,
            "start_time": start_time,
            "end_time": end_time,
            "text": seg
        })

        pointer = pointer_end + 1
        seg_id += 1

        if pointer >= len(sentences):
            break

    output = {
        "episode_id": data["episode_id"],
        "model": "TextTiling",
        "segments": seg_list
    }

    with open(os.path.join(OUT_DIR, file), "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

print("✅ TextTiling baseline completed safely")