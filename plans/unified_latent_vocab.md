# Latent Vocabulary Implementation Plan

## Goal
Optimize memory usage and parameter count by introducing a **Shared Latent Vocabulary Mapping**.
Instead of having multiple independent embedding matrices of size `(vocab_size, model_dim)` (specifically `wte`, `lm_head`, and multiple `value_embeds`), we will share a single mapping from `vocab_size` to `latent_size`.

Specific targets from user:
-   **Vocab Size**: 32,768
-   **Latent Size**: 512
-   **Model Dim**: 384
-   **Value Embedding Layers**: Use `512 x 384` internal weights, taking latent feature vectors as input.
-   **Output Mapping**: Reuse the same Latent->Vocab matrix for the final projection.

## Architectural Changes

### 1. New Configuration
Add `latent_vocab_size` to `GPTConfig`.
```python
@dataclass
class GPTConfig:
    ...
    latent_vocab_size: int = 512
    ...
```

### 2. Shared Mapping Module
Introduce a shared embedding matrix `vocab_map` of shape `(vocab_size, latent_vocab_size)`.
-   **Forward (Input)**: Looks up tokens to produce latent vectors.
    -   `idx (B, T) -> latent_vecs (B, T, latent_vocab_size)`
-   **Backward (Output)**: Projects latent vectors back to vocabulary logits.
    -   `latent_logits (B, T, latent_vocab_size) @ vocab_map.T -> logits (B, T, vocab_size)`

### 3. Updated `GPT` Class

#### Components
*   **`self.vocab_map`**: `nn.Embedding(vocab_size, latent_vocab_size)`.
    *   Acts as the shared "Map from Vocab to Latent Voc".
    *   Size: `32768 x 512` (~16.8M params).
    *   *Note: With latent_size=512, this matrix is significantly smaller than the baseline aggregate.*

*   **`wte` (Token Embedding)**:
    *   Old: `Embedding(vocab_size, model_dim)`
    *   New: `Linear(latent_vocab_size, model_dim, bias=False)`
    *   Input: `latent_vecs` from `vocab_map`.

*   **`value_embeds` (Value Embeddings)**:
    *   Old: `dict(layer_idx: Embedding(vocab_size, model_dim))`
    *   New: `dict(layer_idx: Linear(latent_vocab_size, model_dim, bias=False))`
    *   Input: `latent_vecs` from `vocab_map`.
    *   Size: `512 x 384` (~0.2M params each).

*   **`lm_head` (Language Model Head)**:
    *   Old: `Linear(model_dim, vocab_size)`
    *   New: `Linear(model_dim, latent_vocab_size, bias=False)`
    *   Output: `latent_logits` which are then projected by `vocab_map.T`.

#### Forward Pass
1.  **Input Mapping**:
    ```python
    latent_x = self.vocab_map(idx) # (B, T, 512)
    ```
2.  **Initial Embedding**:
    ```python
    x = self.transformer.wte(latent_x) # (B, T, 384)
    ```
3.  **Value Embeddings**:
    ```python
    ve = self.value_embeds[str(i)](latent_x) # (B, T, 384)
    ```
4.  **LM Head**:
    ```python
    latent_logits = self.lm_head(x) # (B, T, 512)
    logits = latent_logits @ self.vocab_map.weight.T # (B, T, 32768)
    ```

## Parameter Analysis (Current Config)
*   Vocab: 32,768, Latent: 512, Dim: 384.
*   **Existing Baseline**:
    *   WTE: 12.6M
    *   VE (x3): 37.8M
    *   LM Head: 12.6M
    *   **Total Matrix Params**: ~63M
*   **Proposed**:
    *   Vocab Map: 32768 * 512 = 16.8M
    *   WTE Proj: 512 * 384 = 0.2M
    *   VE Proj (x3): 3 * 0.2M = 0.6M
    *   LM Head Proj: 0.2M
    *   **Total Matrix Params**: ~17.8M
    *   *Result*: Massive savings for `dim=384` (63M vs 17.8M) and larger models.

## Checklist
- [ ] Add `latent_vocab_size` to `GPTConfig` in `nanochat/gpt.py`.
- [ ] Refactor `GPT.__init__` to initialize `vocab_map` and update `wte`, `value_embeds`, `lm_head`.
- [ ] Refactor `GPT.forward` to use `latent_x` workflow.
- [ ] Verify shapes and parameter counts in `scripts/base_train_debug.py`.
