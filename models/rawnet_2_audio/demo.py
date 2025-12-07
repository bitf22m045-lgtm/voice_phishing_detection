import subprocess
def demo_cnn():
    subprocess.run(["python", "models/rawnet_2_audio/rawnet_2.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
(demo_cnn())