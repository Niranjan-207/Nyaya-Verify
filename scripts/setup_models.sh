#!/bin/bash
# Description: Setup models script (Ollama only, HF downloaded via Python)

echo "Verifying Ollama installation..."

if ! command -v ollama &> /dev/null; then
    echo "Error: ollama command not found on system path."
    echo "Please install Ollama from https://ollama.com/download"
    exit 1
fi

echo "Ollama found. Pulling llama3.2:3b..."
ollama pull llama3.2:3b

echo "Model successfully pulled!"
