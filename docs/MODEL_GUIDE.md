# MiniCPM5-1B Model Guide

This guide explains the underlying language model powering this project: **OpenBMB's MiniCPM5-1B**, its architecture, memory characteristics, and reasoning capabilities.

---

## 1. What is MiniCPM5-1B?

**MiniCPM** is a family of small, highly efficient open-source language models developed by **OpenBMB** and **Tsinghua University**. 

MiniCPM5-1B has approximately **1 billion parameters** and is built on modern LLaMA architecture standards (RMSNorm, SwiGLU activations, RoPE rotary position embeddings). Despite its compact size, it demonstrates strong reasoning, mathematical derivation, and instruction-following capabilities that rival significantly larger 3B–7B models.

### Key Specifications

| Property | Value |
| :--- | :--- |
| **Developer** | OpenBMB / Tsinghua University |
| **Model Repository** | `openbmb/MiniCPM5-1B` |
| **Parameter Count** | ~1.2 Billion |
| **Underlying Architecture** | LLaMA-style Decoder-Only Causal LM |
| **Context Window** | Up to 4,096 tokens (configured safely to 3,072 in localAI) |
| **Precision Options** | `bfloat16`, `float16`, `int8`, `int4` |
| **Special Capabilities** | Step-by-step Thinking Mode (`<think> ... </think>`) |

---

## 2. Why a 1B Model for Local Hosting?

When hosting AI locally on personal laptops and desktops, hardware constraints are the primary barrier. Many consumer laptops come with discrete GPUs with 4 GB of VRAM (such as the NVIDIA GTX 1650, RTX 3050 mobile, etc.).

A 7B or 8B model (like Llama-3-8B or Mistral-7B) requires:
- ~16 GB VRAM in 16-bit precision.
- ~5.5 GB VRAM even when heavily quantized to 4-bit, causing out-of-memory errors on 4 GB GPUs.

In contrast, **MiniCPM5-1B**:
- Requires only **~2.2 GB of VRAM** in half-precision (`float16` or `bfloat16`).
- Leaves **1.0–1.5 GB of free headroom** for the KV Cache, system processes, and display drivers.
- Delivers rapid token generation (20–40 tokens/sec on consumer GPUs) without requiring cloud servers.

---

## 3. Precision Formats: bfloat16 vs. float16

| Precision | Bits per Parameter | Memory Footprint | Hardware Compatibility | Characteristics |
| :--- | :--- | :--- | :--- | :--- |
| **FP32** | 32 bits | ~4.4 GB | Universal (CPU/GPU) | Full precision; too heavy for 4 GB GPUs. |
| **FP16** | 16 bits | ~2.2 GB | Turing (GTX 1650, RTX 20xx) + newer | High precision, standard tensor core support. |
| **BF16** | 16 bits | ~2.2 GB | Ampere (RTX 30xx) + newer; PyTorch emulated on Turing | Wider dynamic range, prevents numerical overflow during long reasoning chains. |
| **INT4** | 4 bits | ~800 MB | GPU with `bitsandbytes` | Maximum memory savings, slight drop in fine reasoning. |

In `engine.py`, the engine auto-negotiates:
```python
if torch.cuda.is_available():
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
else:
    torch_dtype = torch.float32
```

---

## 4. Deep Reasoning: Thinking Mode

MiniCPM5-1B supports deep reasoning via structured thinking blocks, similar to models like DeepSeek-R1.

### How it Works
When `thinking_mode=True` is enabled in the tokenizer chat template:
```python
inputs = self.tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    enable_thinking=True,
    return_tensors="pt"
)
```
The model pre-populates `<think>\n` in the generation prompt. The model then generates its internal step-by-step reasoning chain before outputting the final answer:

```
<think>
To calculate the compound interest, we use the formula A = P(1 + r/n)^(nt).
Here, Principal P = 1000, rate r = 0.05, n = 1, t = 3.
Step 1: Calculate 1 + r/n = 1.05.
Step 2: 1.05^3 = 1.157625.
Step 3: 1000 * 1.157625 = 1157.63.
</think>
The final accrued amount after 3 years is $1,157.63.
```

### Dynamic Temperature Control
- **Thinking Mode (`temperature = 0.8`)**: A slightly higher temperature encourages exploration of intermediate problem-solving steps, preventing the model from collapsing into overly concise, premature conclusions.
- **Fast Mode (`temperature = 0.4`)**: A lower temperature keeps answers direct, deterministic, and factual without producing `<think>` tags.

### Repetition Penalty (`repetition_penalty = 1.1`)
Smaller parameter models (under 3B) are susceptible to token looping when reasoning across multiple steps. Setting `repetition_penalty = 1.1` penalizes recently generated tokens, ensuring the model progresses logically rather than repeating steps.
