# Layer Doubling Implementation Plan

## Goal
Add functionality to the `GPT` class in `nanochat/gpt.py` to dynamically double the number of transformer layers during runtime. This is often used for "progressive growing" or initializing a deeper model from a shallower trained one (e.g., initializing a 12-layer model from a 6-layer model).

## User Review Required
> [!NOTE]
> This function performs a deep copy of the weights. This increases memory usage immediately. Ensure sufficient CPU/GPU memory is available.

## Proposed Changes

### 1. `nanochat/gpt.py`

#### `GPT` Class
Add a new method `double_layers(self)` to the `GPT` class.

**Logic:**
1.  **Import**: Ensure `copy` is imported (for `deepcopy`).
2.  **Clone**: Iterate through `self.transformer.h` and create deep copies of existing blocks.
3.  **Append**: Extend `self.transformer.h` with the new cloned blocks.
    - *Note*: `nn.ModuleList` supports `.append()` or `.extend()`, which automatically registers the new submodules.
4.  **Update Config**: Update `self.config.n_layer` to the new total.
5.  **Update Window Sizes**: Re-compute or extend `self.window_sizes` to match the new depth.
    - Since `_compute_window_sizes` relies on `config.n_layer` and the tiling pattern, calling it again after updating `config.n_layer` is the safest and most correct approach.
6.  **Re-index**: (Optional but good practice) Update `layer_idx` inside the new blocks if the `Block` class stores it.
    - Looking at existing code: `Block` receives `layer_idx` in `__init__`. It stores `self.attn.layer_idx`.
    - **Crucial**: We must iterate over the new blocks and update their `layer_idx` (and their internal components like `attn.layer_idx`).

**Draft Code:**
```python
    def double_layers(self):
        """
        Double the number of layers by cloning existing blocks and appending them.
        Useful for progressive growth (e.g. initializing depth 12 from depth 6).
        """
        import copy
        
        # 1. Clone existing blocks
        # We perform deepcopy to get independent weights
        new_blocks = [copy.deepcopy(block) for block in self.transformer.h]
        
        # 2. Append new blocks to ModuleList
        # This registers them properly
        start_idx = len(self.transformer.h)
        self.transformer.h.extend(new_blocks)
        
        # 3. Update config
        self.config.n_layer = len(self.transformer.h)
        
        # 4. Fix layer indices in the new blocks
        for i, block in enumerate(self.transformer.h[start_idx:], start=start_idx):
            # If Block/Attn stores layer_idx, we must update it
            if hasattr(block.attn, 'layer_idx'):
                block.attn.layer_idx = i
            
        # 5. Re-compute window sizes for the new depth
        self.window_sizes = self._compute_window_sizes(self.config)
        
        print0(f"Doubled layers from {start_idx} to {self.config.n_layer}")
```

## Verification Plan
1.  **Unit Test**: Create a small script `tests/test_doubling.py` or modify `scripts/base_train_debug.py`.
    - Instantiate small GPT (e.g., 2 layers).
    - Call `double_layers()`.
    - Assert `len(model.transformer.h) == 4`.
    - Assert weight independence: Modify layer 0 weight, ensure layer 2 weight does not change.
    - Run forward pass to check for shape errors (specifically `window_sizes` index errors).
