from fastapi import APIRouter, HTTPException
from services.topic_segment import process_transcript 
import json

router = APIRouter()

@router.post("/segment")
async def segment(transcript_json: dict, with_timestamps: bool = True):
    """ Segment transcription into topics.
    Args:
        transcript (JSON or str): The transcription text or a dictionary containing the transcription.
        with_timestamps (bool): Whether input transcription includes timestamps.
    Returns:
        json: A dict containing the segmented topics.
    """
    if isinstance(transcript_json, str):
        print("Received transcript as string")
        try:
            transcript = json.loads(transcript_json)
        except json.JSONDecodeError as e:
            raise HTTPException(status_code=400, detail= "Invalid JSON format: " + str(e))
    elif isinstance(transcript_json, dict):
        print("Received transcript as dict")
        transcript = transcript_json
    else:
        raise HTTPException(status_code=400, detail="Invalid input type. Expected JSON or string.")
    if not isinstance(transcript, dict):
        raise HTTPException(status_code=400, detail="Transcription must be a dictionary.")
    try:
        result = process_transcript(transcript, with_timestamps)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing transcript: {str(e)}")
    return result