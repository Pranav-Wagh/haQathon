# 🎤 AI Presentation Feedback Assistant (Offline – Qualcomm Hackathon)

This project is an offline AI-powered tool that helps users **improve their presentation skills** by analyzing their **transcript**, **facial expressions**, and **vocal features**. Built using the **Phi-2 LLM** from Hugging Face, the application runs entirely **offline on a Snapdragon X Elite laptop** (no internet required).

---

## ✨ Features

- 📄 **Transcript Analysis**: Understands the structure and tone of the presentation.
- 🎥 **Facial Emotion Logging**: Captures visual expressions like happiness, anger, surprise, etc.
- 🎙️ **Voice Feature Logging**: Logs confidence, clarity, enthusiasm, and pitch from audio.
- 🧠 **LLM-powered Feedback**: Uses Microsoft’s `phi-2` model to provide structured improvement suggestions.
- ✅ **Fully Offline**: All AI models and inference run locally on Snapdragon X Elite – no cloud required.

---

## 🛠 Project Components

### 1. `run_webcam.ipynb`
- Captures webcam feed
- Detects facial emotions using `ResNet + LSTM`
- Records mood data every 5 seconds → saves to `video_logging.txt`

### 2. `audio_logger.py`
- Analyzes microphone input
- Extracts pitch, confidence, clarity, volume, etc.
- Saves summary in `audio_logging.txt`

### 3. `transcript.txt`
- User-written presentation script, paragraph style

### 4. `generate_report.py`
- Loads the three files above
- Prompts the `phi-2` model using a structured instruction
- Saves feedback in `presentation_feedback.txt`

---

## 🧪 Model Used

### 🤖 `phi-2` (from Hugging Face)
- Parameters: ~2.7B
- License: MIT (commercial use allowed)
- Offline-compatible: ✅
- Ideal for: reasoning, instruction following, summarization

> You can download the model via Hugging Face and use it with `transformers`:
```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")
