# Remove Learnable Scaling Parameters Implementation Plan

## Goal
Simplify the `GPT` model validation and training loop by removing the experimental per-layer scaling parameters (`resid_lambdas` and `x0_lambdas`). These parameters were used to scale the residual stream and blend the initial embedding back into the network at each layer.

## User Review Required
> [!WARNING]
> This is a breaking change for checkpoint compatibility. Models trained with these parameters cannot be loaded strictly.

## Proposed Changes

### 1. `nanochat/gpt.py`

#### A. `GPT` Class - `__init__`
- **Remove** initialization of `self.resid_lambdas`.
- **Remove** initialization of `self.x0_lambdas`.

#### B. `GPT` Class - `init_weights`
- **Remove** initialization/fill logic for `self.resid_lambdas`.
- **Remove** initialization/fill logic for `self.x0_lambdas`.

#### C. `GPT` Class - `num_scaling_params`
- **Remove** `scalars` calculation.
- **Update** `total` calculation to exclude `scalars`.
- **Update** return dict to exclude `scalars`.

#### D. `GPT` Class - `estimate_flops`
- **Remove** `self.resid_lambdas.numel()` and `self.x0_lambdas.numel()` from `nparams_exclude`.

#### E. `GPT` Class - `setup_optimizer`
- **Remove** `resid_params` list.
- **Remove** `x0_params` list.
- **Remove** assertion logic checking for `resid_params` and `x0_params`.
- **Remove** optimizer groups for `resid_params` and `x0_params`.

#### F. `GPT` Class - `forward`
- **Remove** `x0 = x` (saving initial embedding).
- **Update** the layer loop:
    - **Remove** the scaling/mixing line: `x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0`.
    - **Keep** only the standard block call: `x = block(x, cos_sin, self.window_sizes[i], kv_cache)`.

## Verification Plan
1.  **Syntax Check**: Ensure `gpt.py` compiles.
2.  **Dry Run**: Run `scripts/base_train_debug.py` to ensure model initializes and runs.
3.  **Forward Pass**: Verify simple forward pass produces valid logits without error.
