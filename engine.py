#!/usr/bin/env python3
"""
MiniCPM Engine (Optimized & Decoupled)

Handles local model loading, token-budgeted memory management,
and real-time streaming token generation via TextIteratorStreamer.
"""

import threading
from typing import Generator, List, Dict, Optional
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer


class MiniCPMEngine:
    """
    Core decoupled inference engine for MiniCPM models.
    Supports token-budgeted history, thinking mode, and streaming generation.
    """

    def __init__(
        self,
        model_path: str = "openbmb/MiniCPM5-1B",
        thinking_mode: bool = False,
        system_prompt: Optional[str] = None,
        max_context_tokens: int = 3072,
    ):
        self.model_path = model_path
        self.thinking_mode = thinking_mode
        self.max_context_tokens = max_context_tokens
        self.history: List[Dict[str, str]] = []

        if system_prompt is None:
            self.system_prompt = (
                "You are an expert problem-solver and assistant. "
                "Always work through problems step by step and explain your reasoning clearly."
            )
        else:
            self.system_prompt = system_prompt

        # Determine optimal precision for local hardware
        if torch.cuda.is_available():
            if torch.cuda.is_bf16_supported():
                torch_dtype = torch.bfloat16
            else:
                torch_dtype = torch.float16
        else:
            torch_dtype = torch.float32

        print(f"Loading {model_path} (dtype={torch_dtype}, thinking={thinking_mode})...")

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        # Load model with SDPA attention for reduced VRAM and faster prefill
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            attn_implementation="sdpa",
        ).eval()

        print(f"Model successfully loaded on {self.model.device}!")

    def count_tokens(self, messages: List[Dict[str, str]]) -> int:
        """Calculate exact token count for a message sequence."""
        encoding = self.tokenizer.apply_chat_template(messages, tokenize=True)
        if isinstance(encoding, dict):
            return len(encoding.get("input_ids", []))
        elif hasattr(encoding, "shape"):
            return encoding.shape[-1]
        elif isinstance(encoding, list):
            return len(encoding)
        return 0

    def manage_history(self):
        """
        Trims conversation history to stay within token budget.
        Trims whole (user, assistant) turn pairs to prevent orphaned messages.
        """
        while len(self.history) > 2:
            current_messages = [{"role": "system", "content": self.system_prompt}] + self.history
            if self.count_tokens(current_messages) <= self.max_context_tokens:
                break
            # Remove the oldest turn pair (user + assistant)
            self.history = self.history[2:]

    def clear_history(self):
        """Clears dialog history and releases unreferenced cached memory."""
        self.history = []
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def generate_stream(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: Optional[float] = None,
    ) -> Generator[str, None, str]:
        """
        Streams generated tokens in real time using TextIteratorStreamer.
        Yields each text fragment as it is produced by the model.
        """
        self.history.append({"role": "user", "content": prompt})
        self.manage_history()

        messages = [{"role": "system", "content": self.system_prompt}] + self.history

        # Tokenize and place directly on target device
        inputs = self.tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            enable_thinking=self.thinking_mode,
            return_tensors="pt",
            return_dict=True,
        ).to(self.model.device)

        if temperature is None:
            temperature = 0.8 if self.thinking_mode else 0.4

        # Streamer for token-by-token emission
        streamer = TextIteratorStreamer(
            self.tokenizer,
            skip_prompt=True,
            skip_special_tokens=True,
        )

        gen_kwargs = {
            **inputs,
            "max_new_tokens": max_new_tokens,
            "do_sample": True,
            "temperature": temperature,
            "top_p": 0.95,
            "repetition_penalty": 1.1,
            "streamer": streamer,
            "pad_token_id": self.tokenizer.eos_token_id,
        }

        # Run model generation in a lightweight background thread
        thread = threading.Thread(target=self.model.generate, kwargs=gen_kwargs, daemon=True)
        thread.start()

        accumulated: List[str] = []
        for text_chunk in streamer:
            accumulated.append(text_chunk)
            yield text_chunk

        thread.join()
        full_response = "".join(accumulated)
        self.history.append({"role": "assistant", "content": full_response})
        return full_response

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 1024,
        temperature: Optional[float] = None,
    ) -> str:
        """Non-streaming convenience wrapper."""
        chunks = list(self.generate_stream(prompt, max_new_tokens=max_new_tokens, temperature=temperature))
        return "".join(chunks)
