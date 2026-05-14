"""Command-line interface for the network log RAG assistant.

Usage:
    python -m src.main ask "were there any failed connections?"
    python -m src.main ask "did the provisioning service have problems?"

Run 'python -m src.ingest' once before using this.
"""

import sys

from src import rag


def main(argv):
    if len(argv) < 2 or argv[0] != "ask":
        print(__doc__)
        return 1

    question = " ".join(argv[1:])
    result = rag.ask(question)

    print("=" * 60)
    print(f"Q: {question}")
    print("=" * 60)
    print(result["answer"])
    print()
    print("sources used:", ", ".join(result["sources"]))
    print("=" * 60)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
