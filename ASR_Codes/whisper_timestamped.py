import whisper
import datetime
import os

# Load the original Whisper large model
model = whisper.load_model("large").to("cuda")  
 
BASE_DIR = os.environ.get("BASE_DIR") 


def format_time(seconds):
    return str(datetime.timedelta(seconds=int(seconds)))

def main():
    for folder_name in os.listdir(BASE_DIR):
        if not folder_name.endswith("-mit-mv"):
            continue
        folder_path = os.path.join(BASE_DIR, folder_name)
        
        for file_name in os.listdir(folder_path):
            if not file_name.endswith("_noisy.wav"):
                continue
            audio_path = os.path.join(folder_path, file_name)

            result = model.transcribe(
                audio_path, 
                language="en", 
                verbose=False,
                task="transcribe"
            )
            transcript = []
            for segment in result['segments']:
                start = format_time(segment['start'])
                end = format_time(segment['end'])
                text = segment['text'].strip()
                transcript.append(f"[{start} - {end}] {text}")
                print(f"[{start} - {end}] {text}")
            # Save the transcript to a file
            full_transcript = "\n".join(transcript)
            out_file = os.path.join(folder_path, f"{folder_name}_timestamped.txt")
            with open(out_file, "w") as f:
                f.write(full_transcript)
            print(f"Transcript saved to {out_file}")
        
if __name__ == "__main__":
    main()
