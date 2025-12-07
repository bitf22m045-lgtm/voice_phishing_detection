import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
# infer_aasist.py
import torch
import torchaudio
import os
from aasist import load_model, MODEL_CONFIG_ARGS
from pathlib import Path

# Hardcoded path to input audio
AUDIO_PATH = "temp/delete.wav"  # change this to your audio
MODEL_PATH = "models/aasist_audio/weights/aasist.pth"     # path to your trained checkpoint
RESULT_FILE = "models/aasist_audio/result.txt"

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess_audio(audio_path, target_samples=64600):
    waveform, sr = torchaudio.load(audio_path)
    waveform = waveform.mean(dim=0, keepdim=True)  # convert to mono
    # Pad or trim to target_samples
    if waveform.size(1) < target_samples:
        pad_size = target_samples - waveform.size(1)
        waveform = torch.nn.functional.pad(waveform, (0, pad_size))
    else:
        waveform = waveform[:, :target_samples]
    return waveform

def main():
    # Load model
    model = load_model(MODEL_PATH, out_score_file_name=None)
    model.to(device)
    model.eval()

    # Preprocess audio
    audio_tensor = preprocess_audio(AUDIO_PATH).to(device)

    # Forward pass
    with torch.no_grad():
        _, output = model(audio_tensor)  # batch size 1
        # Softmax to get probability
        probs = torch.softmax(output, dim=1)
        deepfake_prob = probs[0, 1].item()  # index 1 = 'spoof'

    # Save result
    with open(RESULT_FILE, "w") as f:
        f.write(f"{deepfake_prob:.6f}")


if __name__ == "__main__":
    main()
