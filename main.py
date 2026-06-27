"""
Project Name: AI-Multi-Model-Server
Author: Niroj Kumar Sahoo (nirojkumarsahoo55@gmail.com)
Copyright (c) 2026 Niroj Kumar Sahoo
License: MIT License 
Source: https://github.com/nirojhub/AI-Multi-Model-Server
Description: Main program for a local GPU multi-model server that loads open-source language models
             into GPU memory and exposes REST endpoints for text generation using Qwen and Llama.

"""
# Run using Python 3.12 or less.
# pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cu132
# pip install "fastapi[standard]"

import torch
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from transformers import AutoModelForCausalLM, AutoTokenizer

app = FastAPI(title="Local GPU Multi-Model Server")

# Define request body structure


class PromptRequest(BaseModel):
    prompt: str
    max_tokens: int = 100


# Dictionary to hold our loaded models and tokenizers
local_models = {}


@app.on_event("startup")
def load_models_to_gpu():
    """Loads distinct open-source models directly into GPU VRAM on startup."""
    # Check if a CUDA-capable GPU is available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading models onto target device: {device}")

    # Model 1 Configuration (e.g., Qwen2.5-0.5B) into locally cached directory
    model1_name = "Qwen/Qwen2.5-0.5B-Instruct"
    print(f"Loading {model1_name}...")
    local_models["model_one"] = {
        "tokenizer": AutoTokenizer.from_pretrained(model1_name, cache_dir="./model"),
        "model": AutoModelForCausalLM.from_pretrained(model1_name, cache_dir="./model").to(device)
    }

    # Model 2 Configuration (e.g., Llama-3.2-1B) into locally cached directory
    model2_name = "meta-llama/Llama-3.2-1B"
    print(f"Loading {model2_name}...")
    local_models["model_two"] = {
        "tokenizer": AutoTokenizer.from_pretrained(model2_name, cache_dir="./model"),
        "model": AutoModelForCausalLM.from_pretrained(model2_name, cache_dir="./model").to(device)
    }
    print("All models successfully loaded into GPU memory!")


def generate_text(model_key: str, prompt: str, max_tokens: int) -> str:
    """Helper function to run inference on the selected GPU model."""
    if model_key not in local_models:
        raise HTTPException(status_code=404, detail="Model not initialized")

    components = local_models[model_key]
    tokenizer = components["tokenizer"]
    model = components["model"]

    # Format and push inputs to the exact same GPU device as the model
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)

    with torch.no_grad():
        outputs = model.generate(**inputs, max_new_tokens=max_tokens)

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Endpoint for Model 1


@app.post("/v1/models/qwen")
async def run_qwen(payload: PromptRequest):
    """Endpoint to run inference using the Qwen model."""
    result = generate_text("model_one", payload.prompt, payload.max_tokens)
    return {"model": "Qwen-0.5B", "response": result}

# Endpoint for Model 2


@app.post("/v1/models/llama")
async def run_llama(payload: PromptRequest):
    """Endpoint to run inference using the Llama model."""
    result = generate_text("model_two", payload.prompt, payload.max_tokens)
    return {"model": "Llama-1B", "response": result}

if __name__ == "__main__":
    # Runs the server locally on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
