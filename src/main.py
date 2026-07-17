"""
CLI entry point. Run with:

    python -m src.main

Provides a simple REPL for manual testing during development. This is not
the "messaging interface" itself (that's a separate concern -- terminal here
is just for fast iteration) but the same Agent class is what any real
interface would call into.
"""

from __future__ import annotations

import argparse
import logging

from src.config import load_config
from src.core.agent import Agent
from src.utils.logging import setup_logging

logger = logging.getLogger(__name__)


def repl(agent: Agent):
    print("ADTC Agent -- type a message, or 'exit' to quit.\n")
    while True:
        try:
            message = input("> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if message.lower() in ("exit", "quit"):
            break
        if not message:
            continue
        response = agent.handle_message(message)
        print(f"\n{response}\n")


def main():
    parser = argparse.ArgumentParser(description="ADTC Agent CLI")
    parser.add_argument("--message", type=str, default=None,
                         help="Single message to process, then exit (non-interactive).")
    args = parser.parse_args()

    config = load_config()
    setup_logging(config.log_dir)

    with Agent(config) as agent:
        if args.message:
            print(agent.handle_message(args.message))
        else:
            repl(agent)


if __name__ == "__main__":
    main()
