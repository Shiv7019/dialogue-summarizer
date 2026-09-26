from pathlib import Path
import os
import re

import torch
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from transformers import T5ForConditionalGeneration, T5Tokenizer


app = FastAPI(
    title="Text Summarizer App",
    description="Text Summarization using T5",
    version="1.0"
)



# Paths


BASE_DIR = Path(__file__).resolve().parent



# Hugging Face Model


MODEL_ID = os.getenv(
    "HF_MODEL_ID",
    "Shiv7019/text-summarizer"
)



# Device


if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


print(f"Using device: {device}")
print(f"Loading model: {MODEL_ID}")


# Load Model

tokenizer = T5Tokenizer.from_pretrained(MODEL_ID)

model = T5ForConditionalGeneration.from_pretrained(
    MODEL_ID
)

model.to(device)
model.eval()


# Templates

templates = Jinja2Templates(
    directory=str(BASE_DIR)
)


# Request Model

class DialogueInput(BaseModel):
    dialogue: str


# Text Cleaning

def clean_data(text: str) -> str:

    text = re.sub(r"\r\n", " ", text)
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"<.*?>", " ", text)

    text = text.strip().lower()

    return text


# Summarization

def summarize_dialogue(dialogue: str) -> str:

    dialogue = clean_data(dialogue)

    inputs = tokenizer(
        dialogue,
        padding="max_length",
        max_length=512,
        truncation=True,
        return_tensors="pt"
    )

    inputs = {
        key: value.to(device)
        for key, value in inputs.items()
    }

    with torch.inference_mode():

        targets = model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=150,
            num_beams=4,
            early_stopping=True
        )

    summary = tokenizer.decode(
        targets[0],
        skip_special_tokens=True
    )

    return summary


# API

@app.post("/summarize/")
async def summarize(dialogue_input: DialogueInput):

    summary = summarize_dialogue(
        dialogue_input.dialogue
    )

    return {
        "summary": summary
    }


# Home Page

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={}
    )



# Health Check
@app.get("/health")
async def health():

    return {
        "status": "ok",
        "device": str(device),
        "model": MODEL_ID
    }