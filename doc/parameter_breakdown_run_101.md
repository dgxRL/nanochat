# Parameter Breakdown for Run 101

This document details the calculation of the total **~73.5M** parameters for the configuration in `runs/run-dgx-101.sh`.

## Configuration
*   **Vocab Size ($V$)**: 32,768
*   **Model Dim ($C$)**: 384
*   **Layers ($L$)**: 6
*   **Heads ($H$)**: 6
*   **Head Dim ($D$)**: 64 (so $H \times D = 384$)
*   **Value Embedding Layers**: 3 (Layers 1, 3, 5)

## 1. Embeddings (The Heavy Part)
This architecture is "embedding heavy" because the Value Embeddings (unique to this model) add a full vocabulary projection at multiple layers.

*   **Token Embedding (`wte`)**: 
    $$V \times C = 32,768 \times 384 = \mathbf{12,582,912}$$
*   **LM Head (`lm_head`)**: 
    $$V \times C = 32,768 \times 384 = \mathbf{12,582,912}$$
*   **Value Embeddings (`value_embeds`)**:
    *   Present on alternating layers (indices 1, 3, 5 for a 6-layer model).
    *   Count: 3 matrices.
    *   Shape: $V \times C$ (since keys/values dim matches model dim here).
    *   Total: 
        $$3 \times 12,582,912 = \mathbf{37,748,736}$$

**Subtotal (Embeddings): ~62.9M**

## 2. Transformer Blocks (The Compute Part)
Each of the 6 layers has:

*   **Attention (`attn`)**:
    *   Query (`c_q`): $C \times C = 384^2 = 147,456$
    *   Key (`c_k`): $C \times C = 147,456$
    *   Value (`c_v`): $C \times C = 147,456$
    *   Proj (`c_proj`): $C \times C = 147,456$
    *   *Bias is False for all linear layers.*
    *   *RMSNorm has 0 learnable params.*
    *   **Per Layer Attn**: $589,824$ params.
*   **MLP (`mlp`)**:
    *   Expansion (`c_fc`): $C \times 4C = 384 \times 1536 = 589,824$
    *   Projection (`c_proj`): $4C \times C = 1536 \times 384 = 589,824$
    *   **Per Layer MLP**: $1,179,648$ params.

**Per Block Total**: 
$$589,824 + 1,179,648 = 1,769,472$$

**All 6 Blocks**: 
$$6 \times 1,769,472 = \mathbf{10,616,832}$$

## 3. Grand Total
$$62,914,560 \text{ (Embeddings)} + 10,616,832 \text{ (Blocks)} = \mathbf{73,531,392}$$

**~73.5 Million Parameters**
