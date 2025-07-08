from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset

# Load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-large")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large")


import torch
device = "cuda" if torch.cuda.is_available() else "cpu"
model.to(device)

# Load librespeech dataset
ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]

# Preprocess audio
inputs = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt")
input_features = inputs.input_features.to(device)

# Transcribe
forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")


predicted_ids = model.generate(
    input_features,
    forced_decoder_ids=forced_decoder_ids
)

# Decode and print result
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print("Transcription:", transcription[0])
