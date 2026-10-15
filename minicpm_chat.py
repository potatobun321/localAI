#!/usr/bin/env python3
"""
MiniCPM Chat Assistant Entrypoint (Backward-Compatible Wrapper)

Delegates to the optimized MiniCPMEngine and CLI interface.
"""

from engine import MiniCPMEngine
from cli import run_cli


class MiniCPMChat(MiniCPMEngine):
    """
    Backward-compatible subclass of MiniCPMEngine that provides
    the legacy .chat() method.
    """

    def chat(self) -> None:
        """Start interactive CLI chat session."""
        run_cli()


if __name__ == "__main__":
    run_cli()
