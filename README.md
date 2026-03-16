# NanoGPT (Minimal GPT Training Demo)

This repository contains a minimal GPT-style language model training demo based on Andrej Karpathy's "nanoGPT" / "minGPT" style tutorial. It showcases how to load a small text dataset, build a simple Bigram language model in PyTorch, and train it to generate text.

> **Note:** This is a learning/demo project and is not intended for production use.
>
> **Status:** Work in progress — the code is actively being developed and not yet complete.

## 📁 Repository Structure

- `input.txt` - A sample dataset (Shakespeare text). The training script will also download this file from the web if it is missing.
- `LLM/gptdev.py` - The main training script (exported from a Colab notebook).
- `LLM/gptdev.ipynb` - The original Colab notebook version.
- `LLM/LICENSE` - Apache 2.0 license.

## ✅ Prerequisites

- Python 3.8+ (3.10/3.11 recommended)
- PyTorch (CPU or CUDA build depending on your machine)
- `requests` (for downloading the dataset)

## 🛠️ Installation

Install the required packages using `pip`:

```bash
pip install torch requests
```

> If you have a CUDA-enabled GPU and want to use it for training, install the appropriate CUDA-enabled PyTorch build from https://pytorch.org.

## ▶️ Running the Training Script

From the root of the repository:

```bash
python LLM/gptdev.py
```

The script will:

1. Download `input.txt` from a public URL (if not already present).
2. Read the text and tokenize it character-by-character.
3. Train a tiny Bigram language model.
4. Periodically print training/validation loss and generate a short sample of generated text.

## 🔍 What’s Inside `gptdev.py`

- A simple character-level tokenizer using a vocabulary of unique characters.
- A `BigramLanguageModel` class implemented in PyTorch.
- Training loop with evaluation and sampling.

## 🧠 Notes / Next Steps

If you want to extend this project, consider:

- Implementing a true transformer model (multi-head self-attention, positional embeddings, etc.).
- Using a larger dataset and longer sequences.
- Adding checkpoint saving/loading.
- Adding command-line arguments for hyperparameters.

---

If you want help understanding or extending this code, just ask! 👇
