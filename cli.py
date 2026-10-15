#!/usr/bin/env python3
"""
MiniCPM Terminal CLI

Lightweight, interactive command-line interface for MiniCPMEngine.
Features real-time streaming, command toggles, and zero UI overhead.
"""

import argparse
import sys
from engine import MiniCPMEngine


def parse_args():
    parser = argparse.ArgumentParser(description="MiniCPM Terminal Chat")
    parser.add_argument(
        "--model",
        type=str,
        default="openbmb/MiniCPM5-1B",
        help="Path or Hugging Face ID of the model",
    )
    parser.add_argument(
        "--thinking",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable deep step-by-step thinking mode (default: True)",
    )
    parser.add_argument(
        "--system",
        type=str,
        default=None,
        help="Custom system prompt",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1024,
        help="Maximum new tokens generated per turn (default: 1024)",
    )
    parser.add_argument(
        "--context-budget",
        type=int,
        default=3072,
        help="Maximum tokens allowed in conversation history (default: 3072)",
    )
    return parser.parse_args()


def run_cli():
    args = parse_args()

    engine = MiniCPMEngine(
        model_path=args.model,
        thinking_mode=args.thinking,
        system_prompt=args.system,
        max_context_tokens=args.context_budget,
    )

    print("\n" + "=" * 62)
    print("  MiniCPM Local Chat (Optimized)")
    print(f"  Mode:     {'Thinking (Reasoning)' if engine.thinking_mode else 'Fast'}")
    print("  Commands: 'exit' | 'clear' | 'mode' (toggle thinking) | 'help'")
    print("=" * 62 + "\n")

    while True:
        try:
            user_input = input("\nYou: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye! 👋")
            break

        if not user_input:
            continue

        cmd = user_input.lower()
        if cmd in ["quit", "exit", "q", "/quit", "/exit"]:
            print("Goodbye! 👋")
            break
        elif cmd in ["clear", "/clear"]:
            engine.clear_history()
            print("Conversation history cleared.")
            continue
        elif cmd in ["mode", "/mode"]:
            engine.thinking_mode = not engine.thinking_mode
            mode_name = "Thinking" if engine.thinking_mode else "Fast"
            print(f"Switched mode to: {mode_name}")
            continue
        elif cmd in ["help", "/help"]:
            print("\nAvailable Commands:")
            print("  clear     - Wipe chat history & free VRAM cache")
            print("  mode      - Toggle between Thinking and Fast mode")
            print("  exit/quit - Exit chat")
            continue

        print("\nMiniCPM: ", end="", flush=True)
        try:
            for chunk in engine.generate_stream(user_input, max_new_tokens=args.max_tokens):
                print(chunk, end="", flush=True)
            print()
        except Exception as e:
            print(f"\n[Error during generation: {e}]")


if __name__ == "__main__":
    run_cli()
