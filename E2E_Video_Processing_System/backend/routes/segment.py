from fastapi import APIRouter, File, UploadFile, HTTPException
from typing import JSON
from services.segment import topic_segment 

router = APIRouter()

async def segment(transcript: JSON, with_timestamps: bool = True):
    """ Segment transcription into topics.
    Args:
        transcript (json): The transcription text to segment.
        with_timestamps (bool): Whether to include timestamps in the segmentation.
    Returns:
        json: A JSON object containing the segmented topics.
    """
    result = topic_segment(transcript, with_timestamps)
    return result