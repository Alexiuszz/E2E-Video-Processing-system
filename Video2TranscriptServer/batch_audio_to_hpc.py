import pandas as pd
import os
import requests
import time
from tqdm import tqdm
import dotenv
dotenv.load_dotenv()

# Configuration
BASE_DIR = os.environ.get("BASE_DIR", ".")
SERVER_URL = os.environ.get("SERVER_URL", "http://0.0.0.0:8080")
FASTAPI_SERVER_URL = f"{SERVER_URL}/transcribe" 
OPENAI_FASTAPI_SERVER_URL = f"{SERVER_URL}/openai-transcribe"
AUDIO_SUFFIXES = ["noisy", "denoised_nr", "noisy_DeepFilterNet2", "denoised_vad"]
def get_openai_transcript(audio_path):
    start_time = time.time()
    try:
        with open(audio_path, "rb") as f:
            print(audio_path)
            files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
            response = requests.post(OPENAI_FASTAPI_SERVER_URL, files=files)

        if response.status_code != 200:
            return {
                "error": "Transcription failed",
                "details": response.text
            }, time.time() - start_time

        response_data = response.json()
        return response_data, time.time() - start_time
    except Exception as e:
        return{
            "error": "Exception occurred",
            "details": str(e)
        }, time.time() - start_time
    
# def get_transcript(audio_path):
#     start_time = time.time()
#     try:
#         with open(audio_path, "rb") as f:
#             print(audio_path)
#             files = {"file": (os.path.basename(audio_path), f, "audio/wav")}
#             response = requests.post(FASTAPI_SERVER_URL, files=files)

#         if response.status_code != 200:
#             return {
#                 "error": "Transcription failed",
#                 "details": response.text
#             }, time.time() - start_time

#         response_data = response.json()
#         return response_data, time.time() - start_time

#     except Exception as e:
#         return {
#             "error": "Exception occurred",
#             "details": str(e)
#         }, time.time() - start_time
        
# def batch_audio_to_hpc2(csv_file, output_dir):
#     df = pd.read_csv(csv_file)

#     clean_metrics = []

#     for audio_file_name in tqdm(df['name'], desc="Getting transcripts", unit="file"):
#         subdir = os.path.join(output_dir, audio_file_name)
#         audio_path = os.path.join(subdir, f"{audio_file_name}_raw.wav")
#         transcript_path = os.path.join(subdir, f"{audio_file_name}_small_clean.txt")

#         # Process clean
#         if(not os.path.exists(transcript_path)):
#             result, exec_time = get_transcript(audio_path)
#             if isinstance(result, dict) and "error" in result:
#                 print(f"[ ERROR] {result['error']}: {result['details']}")
#             else:
#                 with open(transcript_path, "w") as f:
#                     f.write(result['transcript'])
#                 clean_metrics.append({
#                     "name": audio_file_name,
#                     "session_id": result['session_id'],
#                     "audio_file": audio_path,
#                     "transcript_file": transcript_path,
#                     "exec_time": exec_time,

#                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),                })


#     # Save metrics
#     pd.DataFrame(clean_metrics).to_csv(os.path.join(output_dir, 'small_transcription_metrics2.csv'), index=False)

# def batch_audio_to_hpc(csv_file, output_dir):
#     df = pd.read_csv(csv_file)

#     clean_metrics = []
#     noisy_metrics = []

#     for audio_file_name in tqdm(df['name'], desc="Getting transcripts", unit="file"):
#         subdir = os.path.join(output_dir, audio_file_name)
#         cleaned_audio_path = os.path.join(subdir, f"{audio_file_name}_raw.wav")
#         cleaned_transcript_path = os.path.join(subdir, f"{audio_file_name}.txt")
#         noisy_audio_path = os.path.join(subdir, f"{audio_file_name}_noisy.wav")
#         noisy_transcript_path = os.path.join(subdir, f"{audio_file_name}_noisy.txt")

#         # Process clean
#         if(not os.path.exists(cleaned_transcript_path)):
#             cleaned_result, clean_time = get_transcript(cleaned_audio_path)
#             if isinstance(cleaned_result, dict) and "error" in cleaned_result:
#                 print(f"[CLEAN ERROR] {cleaned_result['error']}: {cleaned_result['details']}")
#             else:
#                 with open(cleaned_transcript_path, "w") as f:
#                     f.write(cleaned_result['transcript'])
#                 clean_metrics.append({
#                     "name": audio_file_name,
#                     "session_id": cleaned_result['session_id'],
#                     "audio_file": cleaned_audio_path,
#                     "transcript_file": cleaned_transcript_path,
#                     "exec_time": clean_time,
#                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#                 })

#         # Process noisy
#         if(not os.path.exists(noisy_transcript_path)):
#             noisy_result, noisy_time = get_transcript(noisy_audio_path)
#             if isinstance(noisy_result, dict) and "error" in noisy_result:
#                 print(f"[NOISY ERROR] {noisy_result['error']}: {noisy_result['details']}")
#             else:
#                 with open(noisy_transcript_path, "w") as f:
#                     f.write(noisy_result['transcript'])
#                 noisy_metrics.append({
#                     "name": audio_file_name,
#                     "session_id": noisy_result['session_id'],
#                     "audio_file": noisy_audio_path,
#                     "transcript_file": noisy_transcript_path,
#                     "exec_time": noisy_time,
#                     "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
#                 })

#     # Save metrics
#     pd.DataFrame(clean_metrics).to_csv(os.path.join(output_dir, 'clean_transcription_metrics.csv'), index=False)
#     pd.DataFrame(noisy_metrics).to_csv(os.path.join(output_dir, 'noisy_transcription_metrics.csv'), index=False)

def batch_audio_to_openai(csv_file, output_dir):
    df = pd.read_csv(csv_file)
    
    metrics = []
    
    for audio_file_name in tqdm(df['name'], desc="Getting transcripts", unit="file"):
        subdir = os.path.join(output_dir, audio_file_name)
        for suffix in AUDIO_SUFFIXES:
            audio_path = os.path.join(subdir, f"{audio_file_name}_{suffix}.wav")
            openai_transcript_path = os.path.join(subdir, f"{audio_file_name}_{suffix}_openai.txt")
            
            # Check if the OpenAI transcript already exists 
            if not os.path.exists(openai_transcript_path):
                # Call the OpenAI transcription API
                openai_result, exec_time = get_openai_transcript(audio_path)
                if isinstance(openai_result, dict) and "error" in openai_result:
                    print(f"[OPENAI ERROR] {openai_result['error']}: {openai_result['details']}")
                else:   
                    with open(openai_transcript_path, "w") as f:
                        f.write(openai_result['transcript'])
                    metrics.append({
                        "name": audio_file_name,
                        "input_file": audio_path,
                        "model": "openai",
                        "clean_type": suffix,
                        "out_file": openai_transcript_path,
                        "exec_time_sec": exec_time,
                        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                    })
    pd.DataFrame(metrics).to_csv(os.path.join(output_dir, 'openai_transcription_metrics.csv'), index=False)
        
if __name__ == "__main__":
    input_csv = os.path.join(BASE_DIR, 'video_paths.csv')
    batch_audio_to_openai(input_csv, BASE_DIR)