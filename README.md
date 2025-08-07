# Evaluating Automatic Speech Recognition Models for Semantic Topic Segmentation in Educational Video Retrieval

This project aims to explore the trade-off between accuracy and processing time of ASR models and to develop a robust topic segmentation algorithm. It also develops a FastAPI web-based MVP.

## Table of Contents
- [ASR Evaluation](#asr-evaluation)
  - [To get started clone the repo](#to-get-started-clone-the-repo)
  - [Dataprocessing and Audio Enhancement](#dataprocessing-and-audio-enhancement)
  - [Batch Transcription](#batch-transcription)
  - [Get WER](#get-wer)
- [Topic Segmentation](#topic-segmentation)
  - [Dataset Processing](#dataset-processing)
  - [Run Evaluation Script](#run-evaluation-script-in-mainpy)
- [FastAPI MVP](#fastapi-mvp)
- [Acknowledgements](#acknowledgements)

## ASR Evaluation

The first phase of this project evaluates five ASR models on 34 videos collected from [MIT OpenCourseWare](https://ocw.mit.edu/). The dataset includes manually corrected transcripts, which serve as the ground truth and can be provided on request. The evaluation also explores the effect of audio enhancement techniques on Word Error Rate (WER) and Real-Time Factor (RTF).

### To get started clone the repo:

```bash
git clone https://github.com/Alexiuszz/E2E-Video-Processing-system.git
```

### Dataprocessing and Audio Enhancement

1. Enter the dataset processing directory:

```bash
cd DatasetProcessing
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Convert video to audio and create CSV metadata:

```bash
python helper/create_csv_from_dir.py
python video2audio.py
```

4. Apply audio enhancement:

```bash
python audio_enhancement.py
```

### Batch Transcription

1. Navigate to the ASR batch scripts directory:

```bash
cd ../ASR/batch_scripts
```

2. Create a virtual environment and install dependencies:

```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

3. Set environment variables in a `.env` file:

```env
OPENAI_API_KEY=your_openai_key
BASE_DIR=/path/to/your/log/output
```

4. Run any of the batch transcription scripts:

```bash
python whisper_batch.py  
```

### Get WER 

To evaluate Word Error Rate:

- For local models:

```bash
python WER_hpc.py
```

- For OpenAI API:

```bash
python WER_hpc.py
```

## Topic Segmentation

The second phase of the project focuses on developing a robust topic segmentation algorithm, evaluated against three benchmark datasets:

- [AMI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/download/)
- [ICSI Meeting Corpus](https://groups.inf.ed.ac.uk/ami/icsi/download/)
- [YTSeg YouTube Corpus](https://huggingface.co/datasets/retkowski/ytseg/tree/main/data/partitions)

Baseline comparison models include:

- Random segmentation
- Even segmentation
- [Solbiati et al.](https://arxiv.org/abs/2106.12978) (unsupervised segmentation)

To begin:

```bash
cd Segmentation
```

### Dataset Processing

The YTSeg dataset must be preprocessed before running the evaluation:

```bash
python3 datasets_/ytseg_data_preparation.py --input_dir "/path/to/raw/dataset/directory" --output_dir "/path/to/clean/dataset/directory"
```

### Run Evaluation Script in `main.py`

```bash
python main.py --model <model_name> --dataset <dataset_name> [--test_size <num_samples>]
```

- `--model`: Segmentation model. Options: `random`, `bertseg`, `default`, `simple`, `even`
- `--dataset`: Dataset to use. Options: `ytseg`, `ami`, `icsi`
- `--test_size`: (Optional) Limit number of samples

**Example:**

```bash
python main.py --model bertseg --dataset ytseg --test_size 10
```

## FastAPI MVP

Instructions for running the FastAPI MVP can be found in the [README.md of the FastAPI directory](https://github.com/Alexiuszz/E2E-Video-Processing-system/blob/main/E2E_Video_Processing_System/README.md).

## Acknowledgements

- [OpenAI Whisper](https://github.com/openai/whisper)
- [NVIDIA NeMo](https://docs.nvidia.com/nemo-framework/user-guide/latest/installation.html)
- [FFmpeg](https://ffmpeg.org/)
