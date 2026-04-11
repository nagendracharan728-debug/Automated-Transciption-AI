import os
import librosa
import soundfile as sf
import numpy as np
from pydub import AudioSegment, effects, silence
import noisereduce as nr
import matplotlib.pyplot as plt

RAW_DIR = "audio"
CLEAN_DIR = "cleaned_audio"
PLOT_DIR = "plots"

os.makedirs(CLEAN_DIR, exist_ok=True)
os.makedirs(PLOT_DIR, exist_ok=True)

def compute_snr(clean, noisy):
    noise = noisy - clean
    return 10 * np.log10(np.sum(clean**2) / (np.sum(noise**2) + 1e-9))

for file in os.listdir(RAW_DIR):
    if not file.lower().endswith((".wav", ".mp3", ".m4a", ".flac")):
        continue

    print(f"\nProcessing {file}...")

    audio_path = os.path.join(RAW_DIR, file)

    
    y, sr = librosa.load(audio_path, sr=16000, mono=True)

    
    original = y.copy()

    
    y_reduced = nr.reduce_noise(y=y, sr=sr)

    
    temp_path = os.path.join(CLEAN_DIR, f"{file}_temp.wav")
    sf.write(temp_path, y_reduced, sr, subtype="PCM_16")

    
    audio = AudioSegment.from_wav(temp_path)

    
    audio = effects.normalize(audio)

   
    chunks = silence.detect_nonsilent(
        audio,
        min_silence_len=500,
        silence_thresh=-40
    )

    if chunks:
        processed = AudioSegment.empty()
        for start, end in chunks:
            processed += audio[start:end]
    else:
        processed = audio

    
    final_name = os.path.splitext(file)[0] + ".wav"
    final_path = os.path.join(CLEAN_DIR, final_name)
    processed.export(final_path, format="wav")

    print(f"Saved cleaned file: {final_path}")

    
    clean_wave, _ = librosa.load(final_path, sr=16000)

    snr = compute_snr(clean_wave[:len(original)], original[:len(clean_wave)])
    print(f"SNR improvement approx: {snr:.2f} dB")

   
    plt.figure(figsize=(12,4))
    plt.plot(original, alpha=0.5, label="Before")
    plt.plot(clean_wave, alpha=0.7, label="After")
    plt.legend()
    plt.title(file)
    plot_path = os.path.join(PLOT_DIR, file + "_wave.png")
    plt.savefig(plot_path)
    plt.close()

print("\n All audio cleaned + evaluated.")
