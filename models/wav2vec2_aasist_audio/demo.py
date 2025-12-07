import subprocess
def demo_cnn():
    subprocess.run(["python", "models/wav2vec2_aasist_audio/wav2vec2_aasist.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
(demo_cnn())