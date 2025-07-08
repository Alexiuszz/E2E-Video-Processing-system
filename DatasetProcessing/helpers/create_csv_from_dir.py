import pandas as pd
import os
import subprocess
import dotenv

dotenv.load_dotenv()


BASE_DIR = os.environ.get("BASE_DIR")  

def get_video_duration(filepath):
    try:
        result = subprocess.run(
            [
                "ffprobe", "-v", "error", "-select_streams", "v:0",
                "-show_entries", "format=duration",
                "-of", "default=noprint_wrappers=1:nokey=1",
                filepath
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return float(result.stdout.strip())
    except Exception:
        return None

def get_file_size(filepath):
    try:
        return os.path.getsize(filepath)
    except Exception:
        return None
    

def create_csv_from_dir(directory, output_csv):
    """
    Create a CSV file from the directory structure.
    
    Args:
        directory (str): The root directory to scan.
        output_csv (str): The path to the output CSV file.
    """
    data = []
    
    for root, dirs, files in os.walk(directory):
        for file in files:
            if file.endswith('.mp4'):
                video_path = os.path.join(root, file)
                data.append({
                    'name': os.path.basename(os.path.dirname(video_path)),
                    'video_path': video_path,
                    'duration_mins': get_video_duration(video_path)/60,
                    'file_size_mb' : get_file_size(video_path)/(1024*1024)
                
                })
    
    df = pd.DataFrame(data)
    df.to_csv(output_csv, index=False)
    print(f"CSV created at {output_csv}")
    
if __name__ == "__main__":
    output_csv_path = os.path.join(BASE_DIR, 'video_paths.csv')
    create_csv_from_dir(BASE_DIR, output_csv_path)
    print("CSV file created successfully.")
# This script scans a directory for video files, collects their paths, and saves them in a CSV file.
# It also includes functions to get video duration and file size, which can be extended as needed.
# It uses the pandas library to handle CSV operations and os.walk to traverse the directory structure.
# It is designed to be run as a standalone script.
# It can be used to prepare data for further processing, such as audio extraction or analysis.