# DeepFake Media Detector  

**Enterprise-Grade Multi-Modal Deepfake Detection System for Audio, Video, Image, and Text**  

---

## 📌 Overview  

`deepfake_media_detector` is an enterprise-level system designed to detect deepfake content across multiple media types. This repository focuses on **voice/audio deepfake detection**, enabling organizations to identify and mitigate audio-based fraud and phishing attacks efficiently.  

The system provides:  

- **High-accuracy audio deepfake detection**  
- **Support for multiple SOTA models**  
- **Streamlit-based interactive demonstration**  
- **Enterprise-ready modular architecture**  
- **Extensible for future media modalities**  

---

## 🎯 Features  

- Detects deepfake audio in various formats (wav, mp3, etc.)  
- Multiple pre-trained models with configurable selection  
- Calculates **deepfake probability scores**  
- **Streamlit UI** for easy testing and demonstration  
- Clean modular backend architecture using **PyTorch Lightning**  
- Real-time processing for small-to-medium audio datasets  
- Enterprise-friendly logging and reporting  

---

## 🏗 Architecture  
```
+---------------------------+
| Streamlit UI              |
+------------+--------------+
             |
             v
+---------------------------+
| Model Selection Module    |
+------------+--------------+
             |
             v
+---------------------------+
| Audio Processing Engine   |
| (Feature Extraction)      |
+------------+--------------+
             |
             v
+---------------------------+
| Deepfake Probability Calc |
+---------------------------+
             |
             v
+---------------------------+
| Results Output            |
+---------------------------+
```

---

## 🧠 Supported Audio Models  

| Model Name |
|------------|
| AASIST |
| NES2NET_X |
| WAV2VEC2_AASIST |
| WAV2VEC2_ECAPA |
| WAVLM_ECAPA |
| RAWGAT_ST |
| RAWNET_2 |
| XLSR_SLS |
| HUBERTECAPA |
| TCM_ADD |

> **Note:** Remove `_audio` suffix when calling models in your code.  

---

## ⚙ Environment & Dependencies  

**Python Version:** `>=3.9`  

**Dependencies:**  

```text
einops==0.8.1
fairseq==1.0.0a0
librosa==0.9.2
numpy==1.23.3
pandas==2.2.3
pytorch_lightning==2.3.2
PyYAML==6.0.2
rich==13.9.4
scipy==1.15.1
speechbrain==1.0.0
torch
torchaudio
tqdm==4.67.1
transformers==4.45.2
```
## Create a virtual environment
```
conda create -n deepfake_detector python=3.10 -y
conda activate deepfake_detector
```
## Install dependencies
```
pip install -r requirements.txt
```
## 🚀 Usage

1. Start the Streamlit Demo
    ```
    streamlit run demo.py
    ```
2. Upload an Audio File  
    - Supported formats: wav, mp3, flac  
    - Provide a clear, single-speaker sample
3. Select Model  
    - Choose from the supported models listed above
4. Detect Deepfake  
    - Press the Detect button  
5. The backend calculates the deepfake probability
6. Result is displayed along with model confidence


## 🗂 Directory Structure
```
deepfake_media_detector/
│
├─ models/                  # Pre-trained model directories
│   ├─ aasist_audio/
│   ├─ wav2vec2_aasist_audio/
│   └─ ...
│
├─ utils/                   # Utility scripts (feature extraction, metrics)
├─ demo.py                  # Streamlit demonstration
├─ evaluate.py              # Evaluation script for batch audio
├─ requirements.txt         # Environment dependencies
└─ README.md
```
## 🔬 Evaluation
- Evaluate audio datasets using evaluate.py
- Supports batch scoring and CSV/JSON report generation
- Includes metrics like:
  - Accuracy
  - Precision / Recall
  - Deepfake Probability Distribution

## 🧩 Extensibility
- Add new models: Drop the model in the `models/` folder and register it in the model factory

- Multi-modal support: Future modules for image, video, and text deepfake detection

- Enterprise logging: Integrates with `rich` for console logs and can be extended to file-based logging


## 👥 Contributing

Follow these steps to contribute to the project:

1. **Fork the repository**
2. **Create a feature branch**
```bash
git checkout -b feature-name
````
3. **Commit your changes**
```bash
git commit -m "Add feature"
```
4. **Push to your branch**
```bash
git push origin feature-name
```
5. **Open a Pull Request** on GitHub.
   
---

### Coding Standards

* Follow **PEP8** coding standards for Python code.
* Document all changes clearly.
* Ensure tests pass before submitting a PR.


## 📄 License
This project is licensed under the MIT License — see the LICENSE file for details.

## 📞 Contact
Author: Bushra imran

Email: bitf22m045@pucit.edu.pk

GitHub: [github.com/yourusername/deepfake_media_detector](https://github.com/bitf22m045-lgtm/voice_phishing_detection)

## ⚠️ Disclaimer
This software is intended for research and enterprise security purposes only. Use responsibly. The authors are not liable for misuse in illegal or unethical scenarios.

