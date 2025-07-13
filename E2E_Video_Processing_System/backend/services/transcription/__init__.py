def transcribe(file_path: str, model: str):
    if model == "openai":
        from .openai_api import openai_transcribe
        return openai_transcribe(file_path)
    # elif model == "whisper":
    #     from .whisper_local import transcribe_whisper
    #     return transcribe_whisper(file_path)
    # elif model == "nemo":
    #     from .nemo_parakeet import transcribe_nemo
    #     return transcribe_nemo(file_path)
    else:
        raise ValueError("Unsupported model")