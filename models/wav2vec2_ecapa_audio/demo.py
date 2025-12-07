import subprocess
def demo_cnn():
    subprocess.run(["python", "models/wav2vec2_ecapa_audio/wav2vec2_ecapa.py"]
                   ,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL
        )
(demo_cnn())