
import os

BASE_DIR = os.environ.get("BASE_DIR") 

def main():
    for folder_name in os.listdir(BASE_DIR):
        if not folder_name.endswith("-mit-mv"):
            print(folder_name)
            continue
        folder_path = os.path.join(BASE_DIR, folder_name)        
        for file_name in os.listdir(folder_path):
            if not file_name.endswith("_noisy.wav"):
                continue
            audio_path = os.path.join(folder_path, file_name)

            print(audio_path) 
if __name__ == "__main__":
    main()