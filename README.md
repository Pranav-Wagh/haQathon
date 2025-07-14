# 🎤 PresentPerfect: AI-Driven Insights for Mastering Effective Presentations and Public Speaking

This project is an offline AI-powered tool that helps users **improve their presentation skills** by analyzing their **transcript**, **facial expressions**, and **vocal features**. Built using the **Phi-2 LLM** from Hugging Face, the application runs entirely **offline on a Snapdragon X Elite laptop** (no internet required after model download).

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

### 3. `transcript.txt`
- User-written presentation script, paragraph style

### 4. `generate_report.py`
- Loads the three files above
- Prompts the `phi-2` model using a structured instruction
- Saves feedback in `presentation_feedback.txt`

---

## 🤖 Model: Phi-2 (from Hugging Face)

| Name         | microsoft/phi-2 |
|--------------|------------------|
| Parameters   | ~2.7B            |
| License      | MIT              |
| Offline Use  | ✅ Yes           |
| File Size    | ~1.7 GB          |

---

## 📦 How to Download the Model Locally

1. Create a local folder to store the model:
   ```bash
   mkdir phi-2-local

Download the model using Hugging Face's transformers (only once with internet):

from transformers import AutoModelForCausalLM, AutoTokenizer

model = AutoModelForCausalLM.from_pretrained("microsoft/phi-2")
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-2")

model.save_pretrained("phi-2-local")
tokenizer.save_pretrained("phi-2-local")

Update generate_report.py:

model = AutoModelForCausalLM.from_pretrained("phi-2-local", torch_dtype=torch.float16)
tokenizer = AutoTokenizer.from_pretrained("phi-2-local")
