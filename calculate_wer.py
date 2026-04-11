import os
from jiwer import wer, cer

GT_DIR = "ground_truth"        
PRED_DIR = "transcripts_raw/faster_whisper"
   

def read_txt(path):
    with open(path, "r", encoding="utf-8") as f:
        return f.read().lower().strip()

results = []

for gt_file in os.listdir(GT_DIR):

    if not gt_file.endswith(".txt"):
        continue

    base = gt_file.replace(".txt","")
    pred_file = base + ".txt"

    gt_path = os.path.join(GT_DIR, gt_file)
    pred_path = os.path.join(PRED_DIR, pred_file)

    if not os.path.exists(pred_path):
        print(f" Missing prediction for {gt_file}")
        continue

    gt_text = read_txt(gt_path)
    pred_text = read_txt(pred_path)

    if not gt_text or not pred_text:
        print(f"⚠ Empty text in {gt_file}")
        continue

    w = wer(gt_text, pred_text)
    c = cer(gt_text, pred_text)

    print(f"✔ {base}  WER={w:.3f} CER={c:.3f}")
    results.append((base, w, c))

if len(results)==0:
    print("\n No files evaluated.")
else:
    avg_wer = sum(r[1] for r in results)/len(results)
    avg_cer = sum(r[2] for r in results)/len(results)

    print("\n==== FINAL AVERAGE ====")
    print(f"WER={avg_wer:.3f}")
    print(f"CER={avg_cer:.3f}")
