# Text Summarizer

A FastAPI web app for text summarization, powered by a fine-tuned T5 model hosted on Hugging Face Hub.

## Features

- T5-based summarization via Hugging Face Transformers
- Model auto-downloads from Hugging Face Hub on startup — no manual setup
- FastAPI backend, single-page HTML/JS frontend
- REST endpoint: `POST /summarize/`
- Health check: `GET /health`
- Beam search generation (4 beams)
- Automatic device selection: MPS → CUDA → CPU

## Project Structure

```
Text-Summarizer/
├── app.py
├── index.html
├── requirements.txt
└── .gitignore
```

## Setup

```bash
git clone https://github.com/Shiv7019/YOUR-REPOSITORY-NAME.git
cd YOUR-REPOSITORY-NAME

python -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate

pip install -r requirements.txt
```

Run:

```bash
uvicorn app:app --reload
```

Visit `http://127.0.0.1:8000` — interactive docs at `/docs`.

On first run, the app pulls model weights from Hugging Face Hub (`Shiv7019/text-summarizer`). This takes a minute or two depending on your connection; subsequent runs use the local HF cache.

## Configuration

The model source is set via an environment variable:

```bash
export HF_MODEL_ID="your-username/your-model"   # optional — defaults to Shiv7019/text-summarizer
```

Point it at a local directory instead if you want to run fully offline (see below).

## API

**POST** `/summarize/`

```json
// Request
{ "dialogue": "Your text goes here." }

// Response
{ "summary": "Generated summary goes here." }
```

**GET** `/health`

```json
{ "status": "ok", "device": "cuda", "model": "Shiv7019/text-summarizer" }
```

## Pipeline

Input → cleaning (strip HTML/whitespace, lowercase) → T5 tokenizer (max 512 tokens, padded/truncated) → T5 model → beam search (4 beams, max 150 tokens) → summary.

## Offline / Manual Model

If you'd rather not fetch the model at runtime, the same weights are also available on Google Drive:
[Download](https://drive.google.com/drive/folders/1_2bDekiqrmbC7wrk75jj2AL_piP63oBi?usp=sharing)

Download and extract the folder anywhere locally, then point the app at it:

```bash
export HF_MODEL_ID="/path/to/extracted/model"
```

## Tech Stack

Python · FastAPI · Uvicorn · PyTorch · Transformers · SentencePiece · HTML/CSS/JS

## Limitations

- 512-token input limit, 150-token summary limit
- CPU inference is noticeably slower than GPU/MPS
- First run needs internet access to fetch the model, unless using a local path

## Troubleshooting

| Issue | Fix |
|---|---|
| `uvicorn` not recognized | Activate the venv, `pip install uvicorn`, or run `python -m uvicorn app:app --reload` |
| Model download fails / times out | Check your connection, or switch to the offline model path above |
| Port already in use | `uvicorn app:app --reload --port 8001` |

## Roadmap

- [ ] Summary length controls
- [ ] Docker support
- [ ] Auth
- [ ] Cloud deployment
- [ ] Automated tests

## Contributing

PRs welcome — fork, branch, commit, push, open a pull request.

## License

MIT (or your choice)

## Author

**Shivansh** — AI/ML Developer
[GitHub](https://github.com/Shiv7019) · [LinkedIn](https://www.linkedin.com/in/shivansh-mishra-476455381/)
