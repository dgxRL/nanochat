# Removal of Value Embeddings (VE) Implementation Plan

## Goal
Simplify the `GPT` model in `nanochat/gpt.py` by completely removing the experimental "Value Embeddings" (VE) mechanism. This involves stripping out the `value_embeds` dictionary, gating mechanisms in attention, and all associated helper functions.

## User Review Required
> [!WARNING]
> This is a breaking change for checkpoint compatibility. Models trained with VE cannot be loaded into this new architecture without significant surgery (or ignoring missing keys if strict=False).

## Proposed Changes

### 1. `nanochat/gpt.py`

#### A. Helper Functions
- **Remove** `has_ve(layer_idx, n_layer)` function.

#### B. `CausalSelfAttention` Class
- **Remove** `ve_gate_channels` and `ve_gate` layer initialization in `__init__`.
- **Update** `forward` signature: Remove `ve` argument.
- **Update** `forward` logic: Remove the entire "Value residual" block (lines 93-99) where `gate` is computed and `ve` is added to `v`.

#### C. `Block` Class
- **Update** `forward` signature: Remove `ve` argument.
- **Update** `forward` call to `self.attn`: Stop passing `ve`.

#### D. `GPT` Class
- **Update** `__init__`:
    - Remove `self.value_embeds` module dict.
    - Remove the initialization loop for `value_embeds` (lines ~264).
    - Remove the gating zero-init loop (lines ~268).
    - Remove the BF16 cast for `value_embeds` (lines ~280).
- **Update** `num_scaling_params`:
    - Remove `value_embeds` from the count logic.
- **Update** `estimate_flops`:
    - Remove `value_embeds` from parameter count logic.
- **Update** `setup_optimizer`:
    - Remove `value_embeds_params` and its dedicated optimizer group.
- **Update** `forward`:
    - Remove the `ve` lookup logic inside the layer loop: `ve = self.value_embeds[str(i)](idx) ...`.
    - Update the call to `block(...)`: Stop passing `ve`.

## Verification Plan
1.  **Syntax Check**: Ensure `gpt.py` compiles.
2.  **Dry Run**: Run `scripts/base_train_debug.py` (or a simple forward pass script) to ensure the model initializes and runs without errors.
3.  **Parameter Count**: Verify that the parameter count has decreased by roughly `N_layers/2 * Vocab * Dim` (since VE was on half the layers).
