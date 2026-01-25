# Web UI Documentation (`ui.html`)

A simple, standalone HTML/JS interface for interacting with the Chat model.

## Overview

- **Single File**: Contains HTML, CSS, and vanilla JavaScript in one file.
- **API**: Connects to a backend (presumably serving `engine.py` logic) via `POST /chat/completions`.

## Features
- **Chat Interface**: Typical message history view (User/Assistant).
- **Streaming**: Supports Server-Sent Events (SSE) for streaming tokens (`data: {"token": "..."}`).
- **Slash Commands**:
    - `/temperature <float>`: Adjust sampling temperature.
    - `/topk <int>`: Adjust top-k sampling.
    - `/clear`: Reset conversation.
- **Regeneration**: Click on assistant messages to regenerate.
- **Editing**: Click on user messages to edit and restart conversation from that point.

## JavaScript Logic
- `generateAssistantResponse()`: Sends fetch request, reads stream, updates DOM.
- `handleSlashCommand()`: Client-side command parsing.
