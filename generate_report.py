from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import time

start = time.time()
# Load Phi-2 (must be downloaded already)
model_path = "./phi-2-local"  # or your local path
tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True, torch_dtype=torch.float32)
model.eval()

# Read input files
with open("transcript.txt", "r", encoding="utf-8") as f:
    transcript = f.read()

with open("audio_logging.txt", "r", encoding="utf-8") as f:
    audio_log = f.read()

with open("video_logging.txt", "r", encoding="utf-8") as f:
    video_log = f.read()

# Helper to truncate inputs if too long
def truncate_text(text, max_tokens=700):
    tokens = tokenizer.tokenize(text)
    if len(tokens) > max_tokens:
        return tokenizer.convert_tokens_to_string(tokens[:max_tokens])
    return text

# Truncate each block to fit the 2048 token limit
transcript = truncate_text(transcript, 700)
audio_log = truncate_text(audio_log, 700)
video_log = truncate_text(video_log, 500)

# Build the prompt
prompt = f"""
You are an expert in analyzing presentation quality. Based on the transcript, voice features, and video facial emotion logging, provide a very short feedback report.

Transcript:
{transcript}

Audio Emotion Log:
{audio_log}

Video Emotion Log:
{video_log}

Please generate a very short performance review (around 30-40 words) assessing clarity, emotion, confidence, and areas to improve.
"""

# Tokenize and generate output
inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
with torch.no_grad():
    outputs = model.generate(**inputs, max_length=2048, temperature=0.7, top_p=0.95, do_sample=True)

# Decode and save
result = tokenizer.decode(outputs[0], skip_special_tokens=True)

# Extract only the generated part (remove the prompt if echoed)
if prompt in result:
    result = result.replace(prompt, "").strip()

with open("presentation_feedback.txt", "w", encoding="utf-8") as f:
    f.write(result)

end = time.time()

print(f"Time taken: {end - start:.2f} seconds")

print("✅ Presentation feedback saved to 'presentation_feedback.txt'")
