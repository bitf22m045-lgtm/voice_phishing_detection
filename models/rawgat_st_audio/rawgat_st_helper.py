import torch
import torchaudio
import torch.nn.functional as F
from rawgat_st import RawGAT_ST   # your RawGAT model

# -----------------------------
# CONFIG
# -----------------------------
AUDIO_FILE = "test/RealExample1.wav"
CHECKPOINT_FILE = "checkpoints/rawgat_st.pth"
OUTPUT_FILE = "result.txt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# --------------------------------------------------
# LOAD AUDIO PROPERLY (very important)
# --------------------------------------------------
def load_audio(path):
    wav, sr = torchaudio.load(path)         # shape: [channels, samples]

    # Convert stereo → mono
    if wav.shape[0] > 1:
        wav = wav.mean(dim=0, keepdim=True)

    # Final shape for conv1d → [1, 1, samples]
    wav = wav.unsqueeze(0)

    return wav.to(DEVICE)


# --------------------------------------------------
# LOAD MODEL + CHECKPOINT
# --------------------------------------------------
def load_model():
    model = RawGAT_ST().to(DEVICE)

    print("Loading checkpoint:", CHECKPOINT_FILE)
    checkpoint = torch.load(CHECKPOINT_FILE, map_location=DEVICE)

    # Many checkpoints keep weights inside "model" key
    if "model" in checkpoint:
        checkpoint = checkpoint["model"]

    # Load safely (no strict mismatch)
    model.load_state_dict(checkpoint, strict=False)

    model.eval()
    return model


# --------------------------------------------------
# INFERENCE
# --------------------------------------------------
def predict(model, wav):
    with torch.no_grad():
        out = model(wav)      # shape: [1, 1] → score
        prob = torch.sigmoid(out).item()
    return prob


# --------------------------------------------------
# MAIN
# --------------------------------------------------
if __name__ == "__main__":
    print("Loading audio...")
    waveform = load_audio(AUDIO_FILE)

    print("Loading model...")
    model = load_model()

    print("Running inference...")
    deepfake_prob = predict(model, waveform)

    print(f"Deepfake Probability = {deepfake_prob:.4f}")

    # Save result
    with open(OUTPUT_FILE, "w") as f:
        f.write(f"deepfake_probability: {deepfake_prob}\n")

    print("Saved to result.txt")
