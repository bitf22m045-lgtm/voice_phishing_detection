import streamlit as st
import os, time, importlib, shutil
from PIL import Image
import pandas as pd
from utils.del_module import delete_module

models_dir = './models'
temp_dir = './temp'

# ---------------- CLEAN TEMP ON START ----------------
if "initialized" not in st.session_state:
    if os.path.exists(temp_dir):
        for filename in os.listdir(temp_dir):
            file_path = os.path.join(temp_dir, filename)
            try:
                if os.path.isfile(file_path) or os.path.islink(file_path):
                    os.unlink(file_path)
                elif os.path.isdir(file_path):
                    shutil.rmtree(file_path)
            except Exception as e:
                print(f'Failed to delete {file_path}. Reason: {e}')
    st.session_state.initialized = True

favicon = './assets/icon.png'
st.set_page_config(page_title='DeepSafe - Audio DeepFake Detection', page_icon=favicon, initial_sidebar_state='collapsed')
st.write('<style>div.block-container{padding-top:0rem;}</style>', unsafe_allow_html=True)
st.markdown("""<style>#MainMenu, footer, header {visibility: hidden;}</style>""", unsafe_allow_html=True)

# -----------------------------------------------------------
#                    MODEL LOADING
# -----------------------------------------------------------
models_audio = [m for m in os.listdir(models_dir) if m.endswith("_audio")]

st.write("## DeepSafe : Audio DeepFake Detection")

# -----------------------------------------------------------
#                    AUDIO UPLOADER
# -----------------------------------------------------------
uploaded_file = st.file_uploader(
    "Choose an Audio File",
    type=['wav', 'mp3', 'flac', 'mp4', 'mov']  # keep video formats if audio inside
)

show_real_fake_button = False
file_type = None
model_option = []

if uploaded_file:
    # remove any previous temp items
    for f in ["delete.wav"]:
        fp = os.path.join(temp_dir, f)
        if os.path.exists(fp):
            os.remove(fp)

    audio_path = os.path.join(temp_dir, "delete.wav")

    # Save uploaded audio
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.read())

    st.audio(open(audio_path, "rb").read())

    model_option = st.multiselect(
        "Select Audio DeepFake Models",
        [m[:-6].title() for m in models_audio]    # remove "_audio"
    )

    file_type = "audio"
    show_real_fake_button = True


# -----------------------------------------------------------
#                    RUN DETECTION
# -----------------------------------------------------------
if show_real_fake_button:
    if st.button("Real or Fake? 🎤🤔"):

        # clear previous results
        for model in os.listdir(models_dir):
            for fn in ["result.txt", "result.csv"]:
                fp = os.path.join(models_dir, model, fn)
                if os.path.exists(fp):
                    os.remove(fp)

        st.info("Running Audio DeepFake Detectors...")

        inference_times = []
        probabilities = []

        for model in model_option:
            model_name = (model + "_audio").lower()
            module_path = f"models.{model_name}.demo"

            start_time = time.time()
            importlib.import_module(module_path)
            delete_module(module_path)
            inf_time = round(time.time() - start_time, 5)

            # read model output
            res_path = f"models/{model_name}/result.txt"
            with open(res_path) as f:
                prob = round(float(f.readline()), 5)

            probabilities.append(prob)
            inference_times.append(inf_time)

        # Display results
        st.subheader("DeepFake Detection Stats")
        df_stats = pd.DataFrame(
            [probabilities, inference_times],
            index=["DF Probability", "Inference Time"]
        ).T
        df_stats.index = model_option
        st.table(df_stats)

        # cleanup results
        for model in os.listdir(models_dir):
            for f in ["result.txt", "result.csv"]:
                fp = os.path.join(models_dir, model, f)
                if os.path.exists(fp):
                    os.remove(fp)
