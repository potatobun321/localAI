# localAI

A lightweight, resource-optimized local AI chat assistant powered by OpenBMB's MiniCPM5-1B. Designed for resource-constrained consumer hardware (such as 4 GB VRAM GPUs like the NVIDIA GTX 1650), providing an interactive terminal interface and a micro web interface without external web framework dependencies.

---

## Features

- **Decoupled Architecture**: Shared inference core (`engine.py`) feeding both a terminal CLI and a web interface via a Python token generator.
- **Hardware-Optimized Execution**:
  - Automatic precision selection (`bfloat16` or `float16` based on hardware support).
  - Scaled Dot-Product Attention (`attn_implementation="sdpa"`) for lower peak VRAM and faster prompt evaluation.
  - Eliminated per-turn CUDA synchronization stalls to maximize generation throughput.
- **Token-Budgeted Memory Management**: Enforces a strict context token budget (default 3,072 tokens) rather than arbitrary turn counts, preventing Out-Of-Memory (OOM) errors. Always evicts full `(user, assistant)` dialog pairs.
- **Reasoning Mode**: Supports step-by-step thinking generation (`<think> ... </think>`) with dynamic temperature adjustment (`0.8` for reasoning, `0.4` for direct answers).
- **Real-Time Token Streaming**: Generator-based streaming using Hugging Face's `TextIteratorStreamer`.
- **Zero-Dependency Web Server**: Built with Python's standard library `http.server` (< 5 MB RAM overhead, Server-Sent Events, no Flask/Streamlit/npm).
- **System-Wide Shell Launchers**: Standalone bash scripts that automatically resolve their real location across symlinks.

---

## Project Structure

```
localAI/
├── engine.py              # Core inference engine (token budgeting, streaming generator)
├── cli.py                 # Interactive terminal chat interface
├── web.py                 # Micro web server via Python's standard library (SSE)
├── minicpm                # Standalone CLI bash launcher
├── minicpm-web            # Standalone Web server bash launcher
├── minicpm_chat.py        # Backward-compatible entrypoint
├── requirements.txt       # Core Python dependencies
├── .gitignore             # Git ignore definitions
├── static/
│   └── index.html         # Single-file dark-mode web chat UI
└── docs/                  # Technical documentation and guides
    ├── ARCHITECTURE.md    # System design, thread management, and data flow
    ├── MODEL_GUIDE.md     # MiniCPM5-1B architecture, precision, and reasoning
    ├── DATA_AND_OPTIMIZATIONS.md # Memory profiling, bug fixes, and SDPA benchmarks
    └── LOCAL_HOSTING_GUIDE.md    # Guide to running & hosting local models on a PC
```

---

## Prerequisites

- **Operating System**: Linux, macOS, or Windows (WSL2)
- **Hardware**: NVIDIA GPU with 4 GB+ VRAM recommended (CPU fallback supported)
- **Software**: Python 3.10 to 3.12, Git, NVIDIA Drivers & CUDA Toolkit

---

## Installation

```bash
git clone https://github.com/potatobun321/localAI.git
cd localAI

python3 -m venv minicpm-env
source minicpm-env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## Usage

### 1. Terminal Interface

Run interactive CLI chat:
```bash
./minicpm
```

Commands during chat:
- `mode` : Toggle between Thinking (reasoning) and Fast mode.
- `clear`: Clear conversation history and release cached VRAM.
- `exit` or `quit`: Terminate session.

CLI Options:
```bash
./minicpm --no-thinking         # Fast mode without reasoning tags
./minicpm --max-tokens 512      # Maximum tokens per response
./minicpm --context-budget 2048 # Custom context budget
./minicpm --system "Prompt"     # Custom system instruction
```

### 2. Web Interface

Start the lightweight HTTP server:
```bash
./minicpm-web
```

Navigate to `http://127.0.0.1:8000` in a browser.
- Real-time token streaming via Server-Sent Events (SSE).
- Collapsible thought process blocks for reasoning steps.
- In-browser mode toggling and history clearing.

---

## System-Wide Installation

To execute `minicpm` and `minicpm-web` from any working directory, link them to `~/.local/bin`:

```bash
mkdir -p ~/.local/bin
ln -sf "$(pwd)/minicpm" ~/.local/bin/minicpm
ln -sf "$(pwd)/minicpm-web" ~/.local/bin/minicpm-web
```

Ensure `~/.local/bin` is in your `$PATH`.

---

## Documentation

Detailed technical documentation is available in the `docs/` directory:

- [Architecture Overview](docs/ARCHITECTURE.md): System components, generator threading model, and sequence diagrams.
- [Model Guide](docs/MODEL_GUIDE.md): MiniCPM5-1B architecture, precision formats, and reasoning mechanics.
- [Data Handling & Optimizations](docs/DATA_AND_OPTIMIZATIONS.md): Performance profiling, historical bugs resolved, and benchmark comparisons.
- [Local Hosting Guide](docs/LOCAL_HOSTING_GUIDE.md): Complete guide to local AI deployment on consumer hardware.

---

## License

Distributed under the Apache 2.0 License. See [LICENSE](LICENSE) for details.
