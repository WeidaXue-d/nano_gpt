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

# Development Log - Day 2: Brain Architecture & Generation Pipeline 

## March 3, 2026

### Objectives
* Implement the core logic of the **Bigram Language Model**.
* Establish the **Forward Pass** and **Loss Calculation** mechanics.
* Develop an **Inference Pipeline** to generate text from a raw model.

### Key Achievements
* **Model Construction (`model.py`)**:
    * Defined the `BigramLanguageModel` class inheriting from `nn.Module`.
    * Integrated `nn.Embedding` as a primary lookup table (65x65 matrix) to map character tokens to logits.
    * Implemented **Cross Entropy Loss** (`F.cross_entropy`) to quantify the distance between predictions and ground truth.
* **Autoregressive Generation**:
    * Authored the `generate` function using `F.softmax` for probability distribution and `torch.multinomial` for random sampling.
    * Successfully enabled the model to generate sequences of text based on a starting context.
* **System Integration (`train.py`)**:
    * Modularized the project by importing the model class from `model.py`.
    * Verified the mathematical correctness of the initialization: Observed an initial loss of ~$4.5$ - $4.9$, aligning with the theoretical random guess loss of $-\ln(1/65) \approx 4.17$.

### Engineering Insights & Debugging
* **Reshaping Tensors**: Learned that `F.cross_entropy` requires a 2D input `(N, C)`. Used `.view(B*T, C)` to flatten the batch and time dimensions for compatibility.
* **Variable Scope**: Resolved an `UnboundLocalError` caused by a naming inconsistency between `target` (singular) and `targets` (plural). This highlighted the importance of strict naming conventions in large-scale Python projects.
* **Stochastic Generation**: Observed that the model currently outputs gibberish (e.g., `!no3'IHETgl?`). This confirms the generation pipeline is functional, but the model lacks "knowledge" due to the absence of a training loop.

