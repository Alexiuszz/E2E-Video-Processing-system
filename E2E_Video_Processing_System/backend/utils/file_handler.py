import subprocess

def convert_to_wav(input_path: str, output_path: str):
    subprocess.run([
        "ffmpeg", "-y", "-i", input_path,
        "-ar", "16000", "-ac", "1", output_path
    ], check=True)
    
    