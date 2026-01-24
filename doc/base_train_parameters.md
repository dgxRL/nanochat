# Base Train Script Parameters

The following table lists the input parameters for `scripts/base_train.py`.

| Name | Type | Default | Help |
| :--- | :--- | :--- | :--- |
| `run` | str | `"dummy"` | wandb run name ('dummy' disables wandb logging) |
| `device-type` | str | `""` | cuda\|cpu\|mps (empty = autodetect) |
| `depth` | int | `20` | depth of the Transformer model |
| `aspect-ratio` | int | `64` | model_dim = depth * aspect_ratio |
| `head-dim` | int | `128` | target head dimension for attention |
| `max-seq-len` | int | `2048` | max context length |
| `window-pattern` | str | `"SSSL"` | sliding window pattern tiled across layers: L=full, S=half context (e.g. 'SSL') |
| `num-iterations` | int | `-1` | explicit number of optimization steps (-1 = disable) |
| `target-flops` | float | `-1.0` | calculate num_iterations to reach target_flops (-1 = disable) |
| `target-param-data-ratio` | int | `4` | calculate num_iterations to maintain data:param ratio (Chinchilla=20, -1 = disable) |
| `device-batch-size` | int | `32` | per-device batch size |
| `total-batch-size` | int | `524288` | total batch size in tokens |
| `embedding-lr` | float | `0.3` | learning rate for embedding parameters (Adam) |
| `unembedding-lr` | float | `0.004` | learning rate for unembedding parameters (Adam) |
| `weight-decay` | float | `0.2` | cautious weight decay for the Muon optimizer (for weights) |
| `matrix-lr` | float | `0.02` | learning rate for matrix parameters (Muon) |
| `scalar-lr` | float | `0.5` | learning rate for scalars (resid_lambdas, x0_lambdas) |
| `adam-beta1` | float | `0.8` | Adam beta1 for embedding/unembedding |
| `adam-beta2` | float | `0.95` | Adam beta2 for embedding/unembedding |
| `warmup-ratio` | float | `0.0` | ratio of iterations for LR warmup |
| `warmdown-ratio` | float | `0.4` | ratio of iterations for LR warmdown |
| `final-lr-frac` | float | `0.0` | final LR as fraction of initial LR |
| `resume-from-step` | int | `-1` | resume training from this step (-1 = disable) |
| `eval-every` | int | `250` | evaluate val bpb every N steps (-1 = disable) |
| `eval-tokens` | int | `10485760` | number of tokens to evaluate val loss on (default: 20*524288) |
| `core-metric-every` | int | `2000` | evaluate CORE metric every N steps (-1 = disable) |
| `core-metric-max-per-task` | int | `500` | examples per task for CORE metric |
| `sample-every` | int | `2000` | sample from model every N steps (-1 = disable) |
| `save-every` | int | `-1` | save checkpoints every N steps (-1 = only at end) |
| `model-tag` | str | `None` | override model tag for checkpoint directory name |
