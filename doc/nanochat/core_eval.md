# Core Evaluation Documentation (`core_eval.py`)

Implements the CORE metric evaluation logic (from the DCLM paper).

## Overview

Evaluates the model on various tasks (Multiple Choice, Schema, Language Modeling) by measuring the likelihood of continuations.

## Functions

### `evaluate_task(model, tokenizer, data, device, task_meta)`
Evaluates a task across a dataset.
- Distributes examples across ranks.
- Aggregates results (accuracy).

### `evaluate_example(...)`
Evaluates a single example.
- **Prompt Rendering**: Uses Jinja2 templates (`render_prompts_mc`, `render_prompts_schema`, `render_prompts_lm`) to format the input and few-shot examples.
- **Batching**: Finds common prefixes/suffixes to efficiently batch computation.
- **Forward Pass**: Computes autoregressive loss.
- **Scoring**:
    - **MC/Schema**: Chooses option with lowest average loss per token.
    - **Language Modeling**: Checks if greedy generation matches target.

### Evaluation Logic
- **MC**: Common context, different continuations.
- **Schema**: Different contexts, common continuation.
- **LM**: Prompt -> Continuation.
