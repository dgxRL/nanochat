# NVFP8 Implementation Plan for DGX Spark

## Goal
Significantly reduce memory usage and increase training throughput on DGX Spark (assuming H100/Hopper GPUs) by integrating NVIDIA FP8 (NVFP8) using the **Transformer Engine (TE)** library.

## Targeted Files
- `scripts/base_train.py`: Training loop and Autocast configuration.
- `nanochat/gpt.py`: Model architecture definition.
- `nanochat/adamw.py` (referred to as `awamw.py`): Optimizer.
- `nanochat/muon.py`: Optimizer.

## Proposed Changes

### 1. `nanochat/gpt.py` (Model Architecture)
The core strategy is to replace standard PyTorch layers with Transformer Engine's FP8-aware layers.

*   **Imports**:
    *   Add `import transformer_engine.pytorch as te`.
*   **Layer Replacement**:
    *   **Linear Layers**: Replace `torch.nn.Linear` with `te.Linear`.
        *   `CausalSelfAttention`: `c_q`, `c_k`, `c_v`, `c_proj`.
        *   `MLP`: `c_fc` (expansion), `c_proj` (projection).
        *   `GPT`: `wte` (token embedding), `lm_head` (output/unembedding), `value_embeds`.
            *   *Note*: `te.Linear` handles embedding layers efficiently if configured, but for standard embeddings `te.LayerNorm` + `te.Linear` is the pattern for the head. For `wte`, TE often works best when used in conjunction with `LayerNormLinear` or by keeping it BF16 if it's just a lookup. However, for `lm_head`, `te.Linear` is highly recommended.
    *   **Normalization**: Replace `torch.nn.functional.rms_norm` or `nn.RMSNorm` (if used) with `te.RMSNorm`.
        *   Update `Block` to use `te.RMSNorm` modules instead of functional calls if desirable for kernel fusion, or ensure `te.RMSNorm` is used contextually.
        *   Add explicit `ln_embed` and `ln_f` attributes using `te.RMSNorm` to ensure the backbone is fully TE-compatible.
*   **Initialization & Data Types** (Crucial):
    *   **Embeddings to BF16**: Ensure code explicitly casts `wte` and `value_embeds` (and new `te` layers) to `bfloat16` to serve as the master weight type.
    *   *Reference*: See lines ~294-297 (in original code) where `wte` is cast to `bfloat16`. This MUST be preserved or adapted for `te` layers to avoid falling back to FP32 master weights, which would increase memory usage.
    *   `te.Linear` layers by default might init in FP32; force them to BF16 (master weights) if the device supports it.

### 2. `scripts/base_train.py` (Training Loop)
*   **Imports**:
    *   Ensure `import transformer_engine.pytorch as te` and `from transformer_engine.common.recipe import Format, NVFP4BlockScaling` (or standard FP8 recipe) are present.
*   **Context Manager**:
    *   Wrap the forward pass (and loss computation) in `te.fp8_autocast(enabled=True, fp8_recipe=...)`.
    *   Configure the `fp8_recipe` correctly for stability (e.g., `margin=0`, `interval=1`, `fp8_format=Format.HYBRID` or `E4M3`).
*   **Backward Pass**:
    *   Standard `loss.backward()` works, TE handles the casting internally during backward.

### 3. Optimizers (`nanochat/adamw.py`, `nanochat/muon.py`)
*   **Goal**: Use NVFP8 (or compatible low precision) for optimizer states to minimize memory.
*   **Strategy**:
    *   Modify `AdamW` and `Muon` to store momentum/variance states in `torch.float8_e5m2` (if PyTorch version supports `Float8_e5m2` storage) or `torch.bfloat16`.
    *   **FP8 Storage**: If `torch.float8_e5m2` is available (Requires PyTorch 2.3+ or explicit TE support), cast state tensors (`exp_avg`, `momentum_buffer`) to this dtype.
    *   **Fallback**: If FP8 storage implies too much overhead/complexity to implement manually in these custom kernels, strictly enforce `bfloat16` states (which halves memory vs FP32).
    *   **Implementation**: Check `p.data` dtype. If master weights are BF16, ensure optimizer states match BF16 (or go lower to FP8).
    *   *Note*: `te` layers expose master weights. Optimizers should update these master weights.
    *   **Warning**: Accumulating gradients/updates in FP8 is risky. Usually, we store *states* in FP8 but perform the *update math* in FP32/BF16.

## Execution Steps
1.  **Backup**: Ensure `gpt.py` is backed up.
2.  **Refactor GPT**: Modify `GPT` class to use `te` modules.
3.  **Update Train Script**: Verify `base_train.py` has the correct `te.fp8_autocast` wrap.
4.  **Optimizer Update**: Modify `adamw.py` and `muon.py` to use reduced precision for states.
5.  **Dry Run**: Run a short training loop (e.g., 20 steps) to verify:
    *   No loss divergence (NaNs).
    *   Memory usage reduction (check `nvidia-smi` or logs).
    *   Throughput improvement (tokens/sec).

## Risks & Mitigations
*   **Instability**: FP8 has lower precision.
    *   *Mitigation*: Use `delayed_scaling` recipe or `HYBRID` format (E4M3 for FWD, E5M2 for BWD) provided by TE.
*   **Platform Support**: Requires NVIDIA Hopper (H100) or Ada (L40/RTX 4090) architecture.
    *   *Mitigation*: DGX Spark presumably has H100s. If not, FP8 might fall back to BF16 or fail. Add software guards `if te.fp8_available():`.
