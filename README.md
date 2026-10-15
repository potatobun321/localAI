# localAI — Hosting & Running LLMs Locally on PC

[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x%20SDPA-red.svg)](https://pytorch.org/)
[![Model](https://img.shields.io/badge/Model-MiniCPM5--1B-orange.svg)](https://huggingface.co/openbmb/MiniCPM5-1B)
[![License: Apache-2.0](https://img.shields.io/badge/License-Apache%202.0-green.svg)](LICENSE)

A lightweight, resource-optimized local AI chat assistant powered by **OpenBMB's MiniCPM5-1B**. 

Designed as a hands-on learning project demonstrating **how to run and host modern AI models locally on consumer PCs and laptops** (e.g., 4 GB VRAM GPUs like the NVIDIA GTX 1650) with zero cloud dependencies and zero web framework bloat.

---

## 🌟 Key Features

- **Decoupled Architecture**: Shared inference engine (`engine.py`) powering both an interactive terminal CLI and a local web interface.
- **Hardware-Optimized for 4 GB GPUs**:
  - Automatically negotiates optimal hardware precision (`bfloat16` or `float16`).
  - Native PyTorch Scaled Dot-Product Attention (`attn_implementation="sdpa"`) for lower VRAM and fast prompt evaluation.
  - Eliminated CUDA synchronization stalls to boost inter-turn generation throughput.
- **Token-Budgeted Context Window**: Prevents Out-Of-Memory (OOM) crashes by enforcing strict token budgets rather than arbitrary message counts. Always trims complete `(user, assistant)` dialog pairs.
- **Deep Reasoning (Thinking Mode)**: Native support for step-by-step reasoning chains via `<think> ... </think>` tags, with dynamic sampling temperature control.
- **Real-Time Token Streaming**: True generator-based streaming via Hugging Face's `TextIteratorStreamer`.
- **Zero-Dependency Web Server**: Built purely with Python's standard library `http.server` (< 5 MB RAM, Server-Sent Events, no Flask/Streamlit/npm).
- **System-Wide CLI Access**: Standalone shell launchers that can be symlinked to `~/.local/bin` for instant access from any directory.

---

## 📁 Repository Structure

```
localAI/
├── engine.py              # Core inference engine (token budgeting, streaming generator)
├── cli.py                 # Interactive terminal chat interface
├── web.py                 # Micro web server via Python's standard library (SSE)
├── minicpm                # Standalone CLI bash launcher (resolves symlinks)
├── minicpm-web            # Standalone Web server bash launcher (resolves symlinks)
├── minicpm_chat.py        # Backward-compatible entrypoint
├── requirements.txt       # Minimal Python dependencies
├── .gitignore             # Standard Python / virtual environment ignore rules
├── static/
│   └── index.html         # Single-file modern dark-mode web chat UI
└── docs/                  # Detailed documentation & learning guides
    ├── ARCHITECTURE.md    # System design, decoupled streamer, sequence diagrams
    ├── MODEL_GUIDE.md     # MiniCPM5-1B architecture, precision, thinking mode
    ├── DATA_AND_OPTIMIZATIONS.md # Memory profiling, bugs fixed, SDPA benchmarks
    └── LOCAL_HOSTING_GUIDE.md    # Guide to running & hosting local AI on a PC
```

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **OS**: Linux, macOS, or Windows (via WSL2).
- **Hardware**: NVIDIA GPU with 4 GB+ VRAM recommended (or multi-core CPU).
- **Software**: Python 3.10 to 3.12, Git, and NVIDIA Drivers.

### 2. Clone & Setup
```bash
# Clone the repository
git clone https://github.com/potatobun321/localAI.git
cd localAI

# Create and activate virtual environment
python3 -m venv minicpm-env
source minicpm-env/bin/activate

# Install dependencies
pip install --upgrade pip
pip install -r requirements.txt
```

### 3. Running the Assistant

#### A. Interactive Terminal Chat (Zero UI Overhead)
```bash
./minicpm
```
**In-Chat Commands:**
- `mode` : Toggle between **Thinking** (reasoning) mode and **Fast** mode on the fly.
- `clear`: Clear conversation history and reclaim CUDA cache.
- `exit` or `quit`: Exit chat.

**CLI Flags:**
```bash
./minicpm --no-thinking        # Start directly in fast mode without <think> tags
./minicpm --max-tokens 512     # Cap maximum generated tokens
./minicpm --context-budget 2048 # Set custom context token limit
./minicpm --system "You are a Python programming tutor." # Custom system prompt
```

#### B. Micro Web Interface (< 5 MB RAM Overhead)
```bash
./minicpm-web
```
Open **[http://127.0.0.1:8000](http://127.0.0.1:8000)** in your browser:
- Real-time token streaming via Server-Sent Events (SSE).
- Visual collapsing/formatting of `<think>` reasoning blocks.
- One-click Clear History and Thinking Mode toggles.

---

## 💻 System-Wide Terminal Access

To run `minicpm` or `minicpm-web` from **any folder** on your machine:

```bash
mkdir -p ~/.local/bin
ln -sf "$(pwd)/minicpm" ~/.local/bin/minicpm
ln -sf "$(pwd)/minicpm-web" ~/.local/bin/minicpm-web
```

Now you can simply open any terminal tab and type:
```bash
minicpm
```

---

## 📚 Deep-Dive Documentation

For in-depth educational breakdowns of how local AI hosting works, explore the [`docs/`](docs/) directory:

- [**Architecture Overview**](docs/ARCHITECTURE.md) — System design, thread management, and the `TextIteratorStreamer` generator pattern.
- [**Model Guide (MiniCPM5-1B)**](docs/MODEL_GUIDE.md) — Parameter size, precision comparison (`bfloat16` vs `float16` vs `int4`), and reasoning mechanics.
- [**Data Handling & Optimizations**](docs/DATA_AND_OPTIMIZATIONS.md) — Detailed analysis of bottlenecks identified in the original prototype and how token budgeting and SDPA resolved them.
- [**Local AI Hosting Guide**](docs/LOCAL_HOSTING_GUIDE.md) — Step-by-step handbook on hardware selection, software setup, background daemons, and troubleshooting CUDA OOM.

---

## 🛠️ Tech Stack & Tools

- **Language**: Python 3.12
- **Core ML Framework**: [PyTorch 2.x](https://pytorch.org/) (with SDPA attention kernels)
- **Model Ecosystem**: [Hugging Face Transformers](https://huggingface.co/docs/transformers) & [Accelerate](https://huggingface.co/docs/accelerate)
- **Model**: [OpenBMB MiniCPM5-1B](https://huggingface.co/openbmb/MiniCPM5-1B)
- **Web Protocol**: Server-Sent Events (SSE) via Python Standard Library `http.server`
- **Frontend**: Vanilla HTML5, CSS3 (Modern Dark Theme), and ES6 JavaScript

---

## 📄 License & Acknowledgments

- **Model Weights**: Provided by [OpenBMB](https://github.com/OpenBMB/MiniCPM) under their respective model license.
- **Project Code**: Distributed under the [Apache 2.0 License](LICENSE).
