import sys
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration
import librosa
if len(sys.argv) != 3:
    print("Usage: python whisper_transcribe.py <output_textfile>")
    sys.exit(1)
input_audio_file = sys.argv[1]
output_textfile = sys.argv[2]
#3rd argument is optional for model name, large, medium, small, tiny

model_name = sys.argv[3] if len(sys.argv) > 3 else "large"

switcher_model = {
    "large": "openai/whisper-large",
    "medium": "openai/whisper-medium",
    "small": "openai/whisper-small.en"
}
if model_name not in switcher_model:
    print(f"Invalid model name: {model_name}. Choose from: {', '.join(switcher_model.keys())}")
    sys.exit(1)

# Load model and processor
processor = WhisperProcessor.from_pretrained(switcher_model[model_name])
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.to("cuda")

# Load and preprocess audio

audio, sr = librosa.load(input_audio_file, sr=16000)

# Chunk settings
chunk_duration = 30  # seconds
chunk_samples = chunk_duration * sr

all_text = []

# Process in chunks
for start in range(0, len(audio), chunk_samples):
    end = min(start + chunk_samples, len(audio))
    chunk = audio[start:end]

    # Prepare input features
    inputs = processor(chunk, sampling_rate=sr, return_tensors="pt")
    input_features = inputs.input_features.to("cuda")

    # Generate transcription
    decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")
    predicted_ids = model.generate(input_features=input_features, forced_decoder_ids=decoder_ids)

    text = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    print(f"Chunk {start//chunk_samples + 1}: {text}")
    all_text.append(text)

# Save final transcript
with open(output_textfile, "w") as f:
    f.write("\n".join(all_text))