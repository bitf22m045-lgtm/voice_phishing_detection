import subprocess
def demo_cnn():
    subprocess.run(["python", "models/rawgat_st_audio/rawgat_st.py"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL)
(demo_cnn())