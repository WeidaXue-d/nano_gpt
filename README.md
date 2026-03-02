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

## Development Log

### March 2, 2026: Data Infrastructure Layer
* **Tokenization**: Developed a unique character vocabulary (size: 65) and built `encode`/`decode` mappings.
* **Dataset Management**: Split the 1.1M character corpus into a 90/10 train-validation split.
* **Random Anchor Batching**: Implemented the `get_batch` function using a random anchor mechanism to generate `(B, T)` training tensors.
    * `batch_size (B) = 4`
    * `block_size (T) = 8`

### Engineering Insights
* **Tensor Dimensions**: Mastered the use of `torch.randint` and the requirement for tuple shapes `(batch_size,)` to ensure strict PyTorch type compliance.
* **Autoregressive Logic**: Confirmed the target tensor `y` must be offset by 1 relative to `x`, representing the "future" character the model aims to predict.

---

##  Requirements
* Python 3.12+
* PyTorch 2.10.0+
* NumPy 2.4.2+