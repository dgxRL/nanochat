# DeepSeek Multi-Head Latent Attention (MLA) Implementation Plan

## Goal
Replace the standard `CausalSelfAttention` in `nanochat/gpt.py` with **Multi-Head Latent Attention (MLA)** as introduced in the DeepSeek-V2/V3 architecture. MLA reduces Key-Value (KV) cache memory usage significantly via low-rank compression while maintaining model performance.

## Key Concepts (DeepSeek MLA)
1.  **Low-Rank Key-Value Compression**: Instead of storing full `(K, V)` heads, we project inputs into a low-dimensional latent vector $c_{KV}$.
2.  **Decoupled RoPE**: To support compression without degrading positional information, Rotary Positional Embeddings (RoPE) are applied to a separate "pe" vector that is concatenated with the content vector.
3.  **Low-Rank Query Compression**: Queries are also compressed into a latent vector $c_Q$.

## Proposed Changes

### 1. `nanochat/gpt.py`

#### `GPTConfig` Update
Add parameters to control MLA dimensions.
*   `q_lora_rank` (int, default=1536): Dimension of query latent vector $d_{c,Q}$.
*   `kv_lora_rank` (int, default=512): Dimension of KV latent vector $d_{c,KV}$.
*   `nope_head_dim` (int, default=128): Dimension of content (non-rope) part of the head.
*   `rope_head_dim` (int, default=64): Dimension of the RoPE part of the head.
*   `v_head_dim` (int, default=128): Dimension of the value head.

*Note: Total effective head dimension for attention score is `nope_head_dim + rope_head_dim`.*

#### `CausalSelfAttention` Class Rewrite
We will completely restructure `__init__` and `forward`.

**Initialization (`__init__`)**:
1.  **Down-Projections (Compression)**:
    *   `c_b_q`: Linear($d_{model}$, $d_{c,Q}$) (Bias=False)
    *   `c_b_kv`: Linear($d_{model}$, $d_{c,KV}$)
2.  **Norms**:
    *   `q_ln`: RMSNorm($d_{c,Q}$)
    *   `kv_ln`: RMSNorm($d_{c,KV}$)
3.  **Up-Projections (Generation)**:
    *   `c_out_q`: Linear($d_{c,Q}$, $H \times d_{head\_content}$) (Content part of Q)
    *   `c_out_kv`: Linear($d_{c,KV}$, $H \times (d_{head\_content} + d_{head\_v})$) (Content K and V)
        *   *Note*: DeepSeek usually generates K-content and V from the same latent $c_{KV}$.
4.  **Decoupled RoPE Heads (Bypass)**:
    *   `c_rope_k`: Linear($d_{model}$, $d_{rope\_head}$)
        *   *Wait*: DeepSeek V2 derives $k_{rope}$ from the *compressed* latent usually? No, "The RoPE part... involves an additional projection".
        *   DeepSeek V2 Implementation detail: "For keys, we also use a decoupled RoPE strategy... $k_R = \text{RoPE}(x W_{KR})$".
        *   Wait, standard MLA often derives EVERYTHING from $c_{KV}$ except maybe RoPE strategy.
        *   Let's check code or detailed paper: "we use a separate decoupled learning strategy... $k_{rope}$ is generated from $x$ directly? Or from $c_{KV}$?"
        *   Let's assume **Decoupled RoPE from Input** for simplicity and robustness, or stick to strict paper if verifiable.
        *   *Refinement*: V2 paper says $k_R$ and $q_R$ share a decoupled strategy.
        *   Implementation choice:
            *   `q_rope_proj`: Linear($d_{c,Q}$, $H \times d_{rope}$) (Generated from latent Q)
            *   `k_rope_proj`: Linear($d_{model}$, $d_{rope}$) (Direct from Input? Or Latent? Usually Latent for maximum compression).
            *   Actually, effectively caching $c_{KV}$ is the goal. If $k_R$ depends on $c_{KV}$, we can cache $c_{KV}$ and regenerate. If it depends on $x$, we must cache $k_R$ separately.
            *   **Decision**: Generate $k_{rope}$ from $x$ (input) and cache it separately (or compute on fly if cheaper). DeepSeek V2 explicitly mentions caching the "decoupled RoPE key" part separately is expensive? No, "we can concatenate...".
            *   **Let's use the most standard MLA**:
                *   Q: $Compressed(x) \to [q_{content}, q_{rope}]$.
                *   K: $Compressed(x) \to k_{content}$.
                *   K_RoPE: $x \to k_{rope}$.
                *   Attn: $[q_{content}, q_{rope}] \cdot [k_{content}, k_{rope}]^T$.

**Forward Pass**:
1.  **Query Path**:
    *   $x \to c_Q$ (`c_b_q`) $\to$ Norm $\to$ $[q_{content}, q_{rope}]$.
    *   Apply RoPE to $q_{rope}$.
    *   Concat $q = [q_{content}, q_{rope}]$.
2.  **KV Path**:
    *   $x \to c_{KV}$ (`c_b_kv`) $\to$ Norm.
    *   Generate $k_{content}, v$ from $c_{KV}$.
    *   Generate $k_{rope}$ (from $x$ typically, separate head).
    *   Apply RoPE to $k_{rope}$.
    *   Concat $k = [k_{content}, k_{rope}]$.
3.  **Attention**:
    *   Standard Flash Attention on `q`, `k`, `v`.
    *   *Note*: `v` has dim `v_head_dim`. `q, k` have dim `nope + rope`.

#### KV Cache Implications
*   Standard `kv_cache` stores the *projected* K/V.
*   To fully realize MLA benefits in *inference* (Matrix Absorption), one would cache $c_{KV}$ and absorb $W_{UK}$ into $W_{UQ}$.
*   **For this plan**: We will implement the **training architecture** first. The KV cache will store the "materialized" K/V (concatenated) for compatibility with standard FA kernels.
*   *Optimization*: If the user wants memory savings during training, the smaller intermediate $c_{KV}$ helps, but FA requires materialized input.

## Verification
1.  **Shape Check**: Run a forward pass and verify tensor shapes at each step (Latent $\to$ Up-Proj $\to$ Concat).
2.  **Parameter Count**: MLA should generally have fewer parameters for the same memory footprint, or more parameters for the same cache size (depending on rank).
3.  **Forward Pass**: Ensure loss decreases (basic functional check).

## Files to Modify
*   `nanochat/gpt.py`

## Plan
1.  Add `MLAConfig` (or merge into `GPTConfig`).
2.  Re-implement `CausalSelfAttention`.
3.  Update `GPT` to use the new config.
