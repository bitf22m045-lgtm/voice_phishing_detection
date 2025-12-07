import subprocess
def demo_cnn():
    subprocess.run(["python", "models/aasist_audio/infer_aasist.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
(demo_cnn())