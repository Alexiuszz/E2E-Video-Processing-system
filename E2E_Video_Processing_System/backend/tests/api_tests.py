import os
import pytest
from fastapi.testclient import TestClient
from httpx import AsyncClient
from main import app

client = TestClient(app)

TEST_AUDIO_PATH = "tests/test_audio.wav"

# Fixture for generating a small valid WAV file
@pytest.fixture(scope="session", autouse=True)
def generate_test_audio():
    import numpy as np
    from scipy.io.wavfile import write
    os.makedirs("tests", exist_ok=True)
    sr = 16000  
    duration = 1  
    samples = np.random.uniform(-1, 1, sr * duration).astype(np.float32)
    write(TEST_AUDIO_PATH, sr, samples)
    yield
    os.remove(TEST_AUDIO_PATH)


def test_upload_valid_file():
    with open(TEST_AUDIO_PATH, "rb") as f:
        response = client.post("/upload", files={"file": ("test.wav", f, "audio/wav")})
    assert response.status_code == 200
    assert "file_path" in response.json()


def test_upload_invalid_file_type():
    response = client.post("/upload", files={"file": ("test.txt", b"invalid", "text/plain")})
    assert response.status_code == 400
    assert "Unsupported file type" in response.text


def test_transcribe_missing_path():
    response = client.post("/transcribe?model=whisper")
    assert response.status_code == 422  


def test_transcribe_invalid_model():
    with open(TEST_AUDIO_PATH, "rb") as f:
        upload_res = client.post("/upload", files={"file": ("test.wav", f, "audio/wav")})
        file_path = upload_res.json()["file_path"]
    response = client.post(f"/transcribe?file_path={file_path}&model=invalid")
    assert response.status_code == 200
    assert "Unsupported model specified" in response.text or "error" in response.json()


def test_segment_invalid_input():
    response = client.post("/segment", content="Not JSON", headers={"Content-Type": "application/json"})
    assert response.status_code == 400
    assert "Invalid JSON format" in response.text or "Invalid input type" in response.text
