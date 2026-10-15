# Local AI Hosting Guide

This guide is a complete educational handbook on how to host and run AI language models on a personal PC or laptop without relying on cloud APIs.

---

## 1. Core Principles of Local AI Hosting

Hosting an AI model locally means:
1. **Model Weights** are downloaded to your physical disk (via Hugging Face or local files).
2. **Inference Compute** runs on your physical GPU (via CUDA) or CPU (via AVX/BLAS).
3. **Data Privacy**: No prompts, tokens, or personal data ever leave your laptop.
4. **Offline Capability**: Once weights are cached, the assistant functions with zero internet connection.

---

## 2. Hardware Requirements & Considerations

| Component | Minimum Specification | Recommended Specification |
| :--- | :--- | :--- |
| **GPU** | NVIDIA GPU with **4 GB VRAM** (e.g., GTX 1650, RTX 3050 mobile) | NVIDIA GPU with **6 GB+ VRAM** (RTX 3060, RTX 4060) |
| **CUDA Compute** | Compute Capability 6.1+ (Pascal, Turing, Ampere, Ada) | Compute Capability 7.5+ (Turing or newer) |
| **System RAM** | 8 GB | 16 GB DDR4/DDR5 |
| **Disk Space** | 5 GB SSD storage | Fast NVMe SSD |
| **Operating System**| Linux (Ubuntu, Debian, Fedora, Arch) or Windows via WSL2 | Linux |

> [!NOTE]
> If your machine lacks an NVIDIA GPU, PyTorch will run inference on the **CPU**. For a 1B model, CPU inference generates ~5–12 tokens/sec depending on multi-threading.

---

## 3. Step-by-Step Setup from Scratch

### Step 1: Install System Prerequisites
Ensure Python 3.10–3.12 and Git are installed:
```bash
sudo apt update
sudo apt install python3 python3-venv git
```

Ensure NVIDIA drivers and CUDA are installed:
```bash
nvidia-smi
```

### Step 2: Clone the Repository
```bash
git clone https://github.com/potatobun321/localAI.git
cd localAI
```

### Step 3: Create an Isolated Python Environment
```bash
python3 -m venv minicpm-env
source minicpm-env/bin/activate
```

### Step 4: Install PyTorch with CUDA Support
Visit [pytorch.org](https://pytorch.org) to find the command matching your CUDA driver version, or install:
```bash
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements.txt
```

Verify your GPU is recognized by PyTorch:
```bash
python3 -c "import torch; print('CUDA Available:', torch.cuda.is_available(), '| Device:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')"
```

---

## 4. Running the Model

### Option A: Interactive Terminal Chat
```bash
./minicpm
```
Or with custom flags:
```bash
./minicpm --thinking          # Step-by-step reasoning (default)
./minicpm --no-thinking       # Fast, direct response
./minicpm --max-tokens 512    # Cap generation length
```

### Option B: Micro Web Interface
Launch the zero-dependency web server:
```bash
./minicpm-web
```
Open **`http://127.0.0.1:8000`** in any web browser.

---

## 5. System-Wide Terminal Access

To make `minicpm` and `minicpm-web` accessible from **any terminal tab and directory** on your machine:

```bash
mkdir -p ~/.local/bin
ln -sf "$(pwd)/minicpm" ~/.local/bin/minicpm
ln -sf "$(pwd)/minicpm-web" ~/.local/bin/minicpm-web
```

Ensure `~/.local/bin` is in your `$PATH` (most modern Linux distributions include this by default in `~/.profile` or `~/.bashrc`).

---

## 6. Running as a Background Daemon (Optional)

If you want the web interface to always run in the background like a local service:

```bash
nohup /home/kartikay/.local/bin/minicpm-web > minicpm-server.log 2>&1 &
```

To stop the background server:
```bash
pkill -f "web.py"
```

---

## 7. Troubleshooting & FAQ

### CUDA Out of Memory (CUDA OOM)
- **Symptom**: `torch.cuda.OutOfMemoryError: CUDA out of memory.`
- **Cause**: Context length or generated tokens exceeded physical GPU memory.
- **Remedy**:
  1. Reduce the context budget:
     ```bash
     minicpm --context-budget 2048 --max-tokens 512
     ```
  2. Clear other GPU consumers (e.g., closing heavy browser tabs with hardware acceleration or games).

### Symlink Execution Error
- **Symptom**: `.../bin/minicpm-env/bin/python3: No such file or directory`
- **Remedy**: Ensure the launcher script uses `readlink -f "${BASH_SOURCE[0]}"` to resolve the real script directory across symlinks. (This is already implemented in `minicpm` and `minicpm-web`).
