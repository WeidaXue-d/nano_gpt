# nano_# Nano GPT: From-Scratch Implementation 

A minimal, character-level generative language model based on the **Transformer architecture**, trained on the Tiny Shakespeare dataset.

---

## Project Architecture

The development follows a progressive "build-up" approach:
* **Data Infrastructure**: Raw text processing, tokenization, and batching.
* **Bigram Baseline**: Initial character-level lookup model.
* **Self-Attention**: Implementing the core logic for contextual understanding.
* **Transformer Blocks**: Scaling the architecture with multi-head attention and feed-forward layers.

---

##  Requirements
* Python 3.12+
* PyTorch 2.10.0+
* NumPy 2.4.2+