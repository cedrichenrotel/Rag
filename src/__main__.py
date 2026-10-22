from fire import Fire

from src.rag_cli import RagCli


def main() -> None:

    try:
        Fire(RagCli)
    except KeyboardInterrupt:
        print("[WARNING] Forced stop")
        return


if __name__ == "__main__":
    main()
