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


# Development Log - Day 3: Self-Attention & Positional Encoding 

## March 5, 2026

### Objectives
* Transition from a local Bigram model to a global **Self-Attention** mechanism.
* Implement **Positional Embeddings** to provide the model with spatial awareness.
* Optimize the **Inference Pipeline** to handle sequence length constraints.

### Key Achievements
* **Single Head Attention (`Head` class)**:
    * Implemented the **Query (Q)**, **Key (K)**, and **Value (V)** linear transformations.
    * Developed the **Scaled Dot-Product Attention** formula: $Attention(Q, K, V) = \text{softmax}(\frac{QK^T}{\sqrt{d_k}})V$.
    * Applied a **Lower Triangular Mask (tril)** to prevent the model from "cheating" by looking at future tokens during training.
* **Positional Awareness**:
    * Integrated a `position_embedding_table` to encode the order of characters within the `block_size` window.
    * Combined token embeddings with positional embeddings ($x = \text{tok\_emb} + \text{pos\_emb}$) to create time-aware input vectors.
* **Architectural Refactoring**:
    * Upgraded the model from a direct lookup table to a deep learning pipeline: Embedding -> Attention -> Linear Head.

### Engineering Insights & Debugging
* **Scope & Naming Consistency**: Resolved an `UnboundLocalError` caused by inconsistent variable casing (`X` vs `x`). Standardized all tensor variables to lowercase `x` for clarity.
* **Index Error Handling**: Fixed an `IndexError: index out of range` in the `generate` function. Learned that since positional embeddings are fixed to `block_size`, the input sequence must be cropped using `idx[:, -block_size:]` before being fed into the model.
* **Module Architecture**: Fixed an `AttributeError` by merging redundant class definitions and ensuring the `generate` method was correctly localized within the final `nn.Module` class.

###  Current Status
* **Data Layer**: Complete
* **Attention Mechanism**:  Operational (Single Head)
* **Training Loop**: Functional with Attention
* **Loss Performance**: Initial stochastic loss observed at ~4.2, ready for high-step training.

# Development Log - Day 3: Multi-Head Attention & Block Architecture 
## March 8, 2026

### Objectives
* Scale from Single-Head Attention to **Multi-Head Attention (MHA)** for parallel feature extraction.
* Implement the **Feed Forward Network (FFN)** to introduce non-linear processing per token.
* Establish the **Transformer Block** structure using **Residual Connections**.

### Key Achievements
* **Multi-Head Attention (`MultiHeadAttention` class)**:
    * Orchestrated 4 parallel attention heads, each focusing on different segments of the 32-dimensional embedding space ($head\_size = 8$).
    * Implemented a **Projection Layer** (`self.proj`) to re-integrate multi-head outputs into the residual stream.
* **Feed Forward Evolution**:
    * Built a `FeedForward` module with a hidden layer expansion of $4 \times n\_embd$ and **ReLU activation**.
    * Enabled the model to perform "individual thinking" on top of the "communication" provided by attention.
* **Optimization & Training**:
    * Scaled `batch_size` to **64** and `max_steps` to **10,000**.
    * Successfully dropped **Loss from 4.2 to 2.06**, a significant leap in predictive accuracy.

### 💡Engineering Insights & Debugging
* **Residual Connections ($x = x + f(x)$)**: Learned how adding the original input back to the output of a sub-layer prevents gradient vanishing and allows for deeper networks.
* **Dimensional Alignment**: Carefully managed the tensor shapes across MHA and FFN to ensure $(B, T, C)$ consistency.
* **Emergent Properties**: Observed the model starting to "invent" Shakespearean formatting (e.g., `NAME:`, verse line breaks, and archaic