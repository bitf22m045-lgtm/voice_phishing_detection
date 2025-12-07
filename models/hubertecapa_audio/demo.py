import subprocess
import os

def demo_cnn():
    subprocess.run(
        ["python", "models/hubertecapa_audio/hubert_ecapa.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
    )

demo_cnn()
