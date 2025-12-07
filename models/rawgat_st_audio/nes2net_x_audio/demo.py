import subprocess
def demo_cnn():
    subprocess.run(["python", "models/nes2net_x_audio/nes2net_x.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
(demo_cnn())