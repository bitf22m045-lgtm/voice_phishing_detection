import os
import torch
import torch.nn as nn
import torchaudio
import warnings
import contextlib
import sys
from transformers import HubertModel
from speechbrain.lobes.models.ECAPA_TDNN import ECAPA_TDNN

# --------------------------
# Utility: suppress noisy libs
# --------------------------
class HiddenPrints:
    def __enter__(self):
        self._stdout = sys.stdout
        self._stderr = sys.stderr
        sys.stdout = open(os.devnull, 'w')
        sys.stderr = open(os.devnull, 'w')

    def __exit__(self, exc_type, exc_val, exc_tb):
        try:
            sys.stdout.close()
            sys.stderr.close()
        except Exception:
            pass
        sys.stdout = self._stdout
        sys.stderr = self._stderr

# Silence warnings
warnings.filterwarnings("ignore")

CHECKPOINT_DIR = 'models/hubertecapa_audio/checkpoints'
AUDIO_FILE = 'temp/delete.wav'  # single audio file
RESULT_FILE = 'models/hubertecapa_audio/result.txt'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# Silence all stdout/stderr for noisy imports
with open(os.devnull, "w") as fnull:
    with contextlib.redirect_stdout(fnull), contextlib.redirect_stderr(fnull):
        # Import noisy libraries silently
        import torchaudio
        import transformers
        import speechbrain

# Define model
class HubertEcapa(nn.Module):
    def __init__(self):
        super().__init__()
        self.ssl_model = HubertModel.from_pretrained("facebook/hubert-base-ls960")
        self.ecapa_tdnn = ECAPA_TDNN(768, lin_neurons=192)
        self.classifier = nn.Linear(192, 2)

    def forward(self, x):
        x_ssl_feat = self.ssl_model(x, output_hidden_states=True).last_hidden_state
        x = self.ecapa_tdnn(x_ssl_feat)
        x = self.classifier(x.squeeze(1))
        return x

def load_model(checkpoint_path):
    model = HubertEcapa().to(DEVICE)
    if os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=DEVICE)
        if 'state_dict' in checkpoint:
            model.load_state_dict(checkpoint['state_dict'])
        else:
            model.load_state_dict(checkpoint)
    model.eval()
    return model

def process_audio(file_path):
    waveform, sr = torchaudio.load(file_path)
    if sr != 16000:
        waveform = torchaudio.transforms.Resample(sr, 16000)(waveform)
    waveform = waveform.mean(dim=0, keepdim=True)
    return waveform.to(DEVICE)

def predict(model, audio_tensor):
    with torch.no_grad():
        output = model(audio_tensor)
        prob = torch.softmax(output, dim=1)[:, 1].item()  # deepfake probability
    return prob

def main():
    model_path = os.path.join(CHECKPOINT_DIR, 'hubert_ecapa.ckpt')
    model = load_model(model_path)

    audio_tensor = process_audio(AUDIO_FILE)
    prob = predict(model, audio_tensor)

    # Save to result.txt
    with open(RESULT_FILE, 'w') as f:
        f.write(f"{prob:.4f}\n")

if __name__ == "__main__":
    with HiddenPrints():
        main()
