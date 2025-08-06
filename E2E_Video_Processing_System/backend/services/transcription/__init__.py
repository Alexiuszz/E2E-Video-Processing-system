import requests

def transcribe(file_path: str, model: str):
    if model == "openai":
        from .openai_api import openai_transcribe
        return openai_transcribe(file_path)

    elif model in ["whisper", "nemo"]:
        url = "http://localhost:8080/transcribe" #forwarded port from HPC server

        with open(file_path, "rb") as f:
            files = {"file": (file_path, f, "audio/wav")}  # or use mimetypes.guess_type
            data = {"model": model}

            try:
                response = requests.post(url, files=files, data=data)
                response.raise_for_status()
                return response.json()
            except requests.RequestException as e:
                raise RuntimeError(f"Failed to get transcription from server: {e}")

    else:
        raise ValueError("Unsupported model")
