import subprocess
def demo_cnn():
    subprocess.run(["python", "models/wavlm_ecapa_audio/wavlm_ecapa.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
(demo_cnn())