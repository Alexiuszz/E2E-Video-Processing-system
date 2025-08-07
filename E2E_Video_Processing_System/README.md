
# Lecture Transcription & Topic Segmentation System

This project provides a complete pipeline for processing lecture recordings. It includes media upload, automatic speech recognition (ASR), and semantic topic segmentation, all accessible via a FastAPI backend and a React frontend.

---

## Components

### 1. Backend API Server (FastAPI)
- Accepts media files
- Performs transcription using OpenAI API or forwards to local Whisper/NeMo on HPC
- Segments transcripts into coherent topics
- Returns JSON with timestamps and topics

### 2. HPC Model Server (FastAPI)
- Runs local Whisper (medium) or Nvidia NeMo (Parakeet) for GPU-accelerated transcription
- Accepts multipart audio and model parameters
- Returns transcription output to main API

### 3. Frontend (React + Tailwind CSS)
- Uploads media
- Allows model selection
- Tracks progress through upload → transcribe → segment stages
- Displays segmented transcript with preview and optional export

---

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- CUDA-compatible GPU (for local Whisper/NeMo)
- FFmpeg

---

## Backend Setup

### Main FastAPI API Server

1. Clone the repo:

```bash
git clone https://github.com/your-org/lecture-pipeline.git
cd /backend
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Set environment variables:

Create a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
BASE_DIR=/path/to/your/log/output
```

4. Run the server:

```bash
uvicorn main:app --reload --port 8000
```

### HPC Model Server

Used only for Whisper local and NeMo transcription (offloaded to GPU)

1. Install Whisper and NeMo:

```bash
pip install git+https://github.com/openai/whisper.git
pip install nemo_toolkit['asr']
```

2. Launch server:

```bash
uvicorn model_server:app --host 0.0.0.0 --port 8080
```

Make sure this port is accessible from the main API.

---

## Frontend Setup

1. Go to the frontend directory:

```bash
cd ../frontend
```

2. Install dependencies:

```bash
npm install
```

3. Start the development server:

```bash
npm run dev
```

This runs the app at `http://localhost:5173` by default.

---

## API Overview

### `/upload` (POST)

- Uploads media file
- Returns: `{ "file_path": "/tmp/tmpabc123.wav" }`

### `/transcribe` (POST)

- Params: `file_path`, `model` (e.g., `openai`, `whisper`, `nemo`)
- Returns: `{ "segments": [...] }`

### `/segment` (POST)

- Input: transcript JSON
- Returns: segmented topics with timestamps

---

## Key Features

- Modular FastAPI backend
- Supports Whisper (OpenAI + local) and Nvidia NeMo
- Chunking for large files
- Timestamp alignment and adaptive segmentation
- Lightweight React UI
- Deployable to HPC or cloud VM

---

## Project Structure

```
/backend
  ├── main.py                  # API entry point
  ├── routes/                  # /upload, /transcribe, /segment
  ├── services/                # transcription + topic segmentation
  ├── utils/                   # helper functions (e.g., audio conversion)

/frontend
  ├── App.tsx                  # UI flow
  ├── components/              # FileUpload, Preview, Button, Progress
  ├── utils/api.ts             # Axios client
```

---


---

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [NVIDIA NeMo](https://developer.nvidia.com/nemo)

