# NanoGPT (Minimal GPT Training Demo)

This repository contains a minimal GPT-style language model training demo based on Andrej Karpathy's "nanoGPT" / "minGPT" style tutorial. It showcases how to load a small text dataset, build a Transformer-based language model in PyTorch with multi-head self-attention, and train it to generate coherent text.

> **Note:** This is a learning/demo project and is not intended for production use.
>
> **Status:** Work in progress — the code is actively being developed and not yet complete.

##  Repository Structure

- `input.txt` - A sample dataset (Shakespeare text). The training script will also download this file from the web if it is missing.
- `LLM/gptdev.py` - The main training script (exported from a Colab notebook).
- `LLM/gptdev.ipynb` - The original Colab notebook version.
- `LLM/LICENSE` - Apache 2.0 license.

##  Prerequisites

- Python 3.8+ (3.10/3.11 recommended)
- PyTorch (CPU or CUDA build depending on your machine)
- `requests` (for downloading the dataset)

##  Installation

Install the required packages using `pip`:

```bash
pip install torch requests
```

> If you have a CUDA-enabled GPU and want to use it for training, install the appropriate CUDA-enabled PyTorch build from https://pytorch.org.

##  Running the Training Script

From the root of the repository:

```bash
python LLM/gptdev.py
```

The script will:

1. Download `input.txt` from a public URL (if not already present).
2. Read the text and tokenize it character-by-character.
3. Train a tiny Bigram language model.
4. Periodically print training/validation loss and generate a short sample of generated text.

##  What’s Inside `gptdev.py`

- **Character-level tokenizer** using a vocabulary of unique characters
- **Multi-head self-attention** (`Head` and `MultiHeadAttention` classes) with causal masking
- **Transformer blocks** with 6 layers of attention + feedforward networks
- **FeedForward layers** (linear → ReLU → linear with dropout)
- **Configurable hyperparameters**:
  - `n_embd = 384` — embedding dimensions
  - `n_head = 6` — number of attention heads
  - `n_layer = 6` — number of transformer blocks
  - `block_size = 256` — context length for predictions
  - `dropout = 0.2` — regularization to prevent overfitting
- **Training loop** with periodic loss evaluation and text generation

##  Notes / Next Steps

Potential improvements and extensions:

- Implement positional embeddings for better context awareness
- Add checkpoint saving/loading to resume training
- Use a larger dataset (e.g., full Project Gutenberg texts)
- Add command-line argument parsing for hyperparameters
- Implement layer normalization for more stable training
- Experiment with different optimizer schedules (learning rate warmup, decay, etc.)
- Visualize attention patterns to understand what the model learns
- Fine-tune hyperparameters based on validation loss curves

---

If you want help understanding or extending this code, just ask!
