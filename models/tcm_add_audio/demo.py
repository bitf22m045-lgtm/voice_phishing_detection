import subprocess
def demo_cnn():
    subprocess.run(["python", "models/tcm_add_audio/tcm_add.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
(demo_cnn())