# Tokenizer Module Documentation (`tokenizer.py`)

Handles text-to-token encoding and decoding, supporting special tokens for chat and tool use.

## Overview

Supports two implementations:
1.  **HuggingFace Tokenizer**: Wrapper around `tokenizers` library (GPT-4 style).
2.  **RustBPE Tokenizer**: Custom training wrapper (`rustbpe`) + `tiktoken` for efficient inference.

## Special Tokens
- `<|bos|>`: Beginning of sequence / Document delimiter.
- `<|user_start|>` / `<|user_end|>`: User message delimiters.
- `<|assistant_start|>` / `<|assistant_end|>`: Assistant message markers.
- `<|python_start|>` / `<|python_end|>`: Python code block markers.
- `<|output_start|>` / `<|output_end|>`: Tool output markers.

## Classes

### `RustBPETokenizer` (Preferred)
- **Init**: Loaded from directory (`tokenizer.pkl` containing `tiktoken` encoding).
- **`encode(text)`**: Encodes text to ids. Handles prepend/append special tokens.
- **`render_conversation(conversation)`**: converting a chat list (messages) into a token sequence with masking (mask=1 for assistant tokens to train on).
- **`render_for_completion(conversation)`**: Prepares a conversation for the model to generate the next response.

### `HuggingFaceTokenizer`
- Similar interface, wraps `tokenizers.Tokenizer`. Used mainly for initial compatibility or if `rustbpe` is not available.

## Helper Functions
- `get_tokenizer()`: returns the configured tokenizer instance.
- `get_token_bytes()`: returns the byte length of each token (for `bpb` metric).
