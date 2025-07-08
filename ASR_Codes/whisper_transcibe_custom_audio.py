import sys
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration

if len(sys.argv) != 3:
    print("Usage: python whisper_transcribe.py <output_textfile>")
    sys.exit(1)
input_audio_file = sys.argv[1]
output_textfile = sys.argv[2]

# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")
model.to("cuda")

# Load and preprocess audio
import librosa
audio, sr = librosa.load(input_audio_file, sr=16000)

#Prepare input for whisper
inputs = processor(audio, sampling_rate=sr, return_tensors="pt")
input_features = inputs.input_features.to("cuda")

# Transcribe
decoder_ids  = processor.get_decoder_prompt_ids(language="en", task="transcribe")

# Generate and decode transcription
predicted_ids = model.generate(
    input_features = input_features,
    forced_decoder_ids=decoder_ids
)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]

# Save transcription to output file
with open(output_textfile, "w") as f:
    f.write(transcription)