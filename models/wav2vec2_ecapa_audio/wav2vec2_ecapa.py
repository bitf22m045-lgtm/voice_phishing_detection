import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import librosa
import soundfile as sf
import numpy as np


# ============================================================
# 1. RAWGAT-ST MODEL (MINIMAL WORKING VERSION)
# ============================================================

class SEBlock(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SEBlock, self).__init__()
        self.fc1 = nn.Linear(channel, channel // reduction)
        self.fc2 = nn.Linear(channel // reduction, channel)

    def forward(self, x):
        y = torch.mean(x, dim=-1)
        y = F.relu(self.fc1(y))
        y = torch.sigmoid(self.fc2(y))
        return x * y.unsqueeze(-1)


class RawGATST(nn.Module):
    def __init__(self, num_classes=2):
        super(RawGATST, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),

            nn.Conv1d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
        )

        self.se = SEBlock(64)
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.encoder(x)          # [B, 64, T]
        x = self.se(x)               # SE block
        x = torch.mean(x, dim=-1)    # Global avg pool -> [B, 64]
        x = self.fc(x)               # Logits
        return x


# ============================================================
# 2. LOAD AUDIO
# ============================================================

def load_audio(file_path, target_sr=16000):
    audio, sr = sf.read(file_path)
    if sr != target_sr:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=target_sr)
    if len(audio.shape) > 1:
        audio = np.mean(audio, axis=1)
    audio = audio / (np.max(np.abs(audio)) + 1e-9)
    audio = torch.tensor(audio, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    return audio   # shape [1, 1, T]


# ============================================================
# 3. INFERENCE FUNCTION
# ============================================================

def inference(audio_path, checkpoint_path):
    # Load model
    model = RawGATST(num_classes=2)

    print("Loading checkpoint...")
    # Plain PyTorch load (no weights_only=True)
    ckpt = torch.load(checkpoint_path, map_location="cpu")
    print("Checkpoint loaded.")

    # Lightning checkpoints may store weights under "state_dict" or "model"
    if "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    elif "model" in ckpt:
        ckpt = ckpt["model"]

    # Remove Lightning-specific prefixes if needed
    new_ckpt = {}
    for k, v in ckpt.items():
        if k.startswith("model."):
            new_ckpt[k[6:]] = v  # remove 'model.' prefix
        else:
            new_ckpt[k] = v

    model.load_state_dict(new_ckpt, strict=False)
    model.eval()

    # Load audio
    x = load_audio(audio_path)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    fake_prob = float(probs[1])
    real_prob = float(probs[0])
    return fake_prob, real_prob



# ============================================================
# 4. MAIN
# ============================================================

if __name__ == "__main__":
    audio_path = "temp/delete.wav"
    checkpoint_path = "models/wav2vec2_ecapa_audio/checkpoints/rawgat_st.pth"

    fake_p, real_p = inference(audio_path, checkpoint_path)

    print("\n--- Deepfake Detection Result ---")
    print(f"Real probability     : {real_p:.6f}")
    print(f"Deepfake probability : {fake_p:.6f}")

    # Save to file
    result_path = "models/wav2vec2_ecapa_audio/result.txt"
    os.makedirs(os.path.dirname(result_path), exist_ok=True)
    with open(result_path, "w") as f:
        f.write(f"{fake_p:.6f}\n")

    print(f"\nSaved result to {result_path}\n")
