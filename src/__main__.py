import sys

from fire import Fire

try:
    from src.rag_cli import RagCli
except KeyboardInterrupt:
    print("[WARNING] Forced stop")
    sys.exit()


def main() -> None:

    try:
        Fire(RagCli)
    except KeyboardInterrupt:
        print("[WARNING] Forced stop")
        sys.exit()


if __name__ == "__main__":
    main()
