from pathlib import Path


def list_files(root: str) -> list[Path]:
    """parcour tous les fichiers du dossier data/raw et creer une
    liste de chemin de tous les fichiers dont le suffixce termine
    par: .py, .md, .txt"""

    root_path: Path = Path(root)

    if not root_path.is_dir():
        raise ValueError(f"Not a directory: {root}")

    list_file: list[Path] = []
    for path in root_path.rglob("*"):
        if path.is_file() and path.suffix in (".md", ".py", ".txt"):
            list_file.append(path)

    return list_file


def read_file(file: Path) -> str:
    """Retourne le texte d'un fichier, celui dont on lui
    donne le chemin."""

    return file.read_text(encoding='utf-8')
