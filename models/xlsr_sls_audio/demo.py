import subprocess
def demo_cnn():
    subprocess.run(["python", "models/xlsr_sls_audio/xlsr_sls.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
(demo_cnn())