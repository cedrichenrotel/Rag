from pathlib import Path

from src.chunk import CutChunk
from src.model import Chunk


class ChunkCorpus:
    """Lit le corpus et le decoupe en chunks.

    Recupere les fichiers utiles (.md, .txt, .py), puis fait appel
    a CutChunk pour decouper chaque texte selon son type de fichier.
    """

    def list_files(self, root: str) -> list[Path]:
        """parcour tous les fichiers du dossier data/raw et creer une
        liste de chemin de tous les fichiers dont le suffixce termine
        par: .py, .md, .txt"""

        root_path: Path = Path(root)

        if not root_path.is_dir():
            raise ValueError(f"[WARNING] list_files: Not a directory: {root}")

        list_file: list[Path] = []
        for path in root_path.rglob("*"):
            if path.is_file() and path.suffix in (".md", ".py", ".txt"):
                list_file.append(path)

        return list_file

    def read_file(self, file: Path) -> str:
        """Retourne le texte d'un fichier, celui dont on lui
        donne le chemin."""

        return file.read_text(encoding="utf-8")

    def chunk_corpus(self, root: str, max_chunk_size: int) -> list[Chunk]:
        """retourne une liste de chunk de tous le corpus"""

        path_list: list[Path] = self.list_files(root)
        if not path_list:
            raise FileNotFoundError(
                "[WARNING] chunk_corpus: File (“.md”/.txt'/“.py”) not found"
            )

        cutchunk: CutChunk = CutChunk()
        list_chunks: list[Chunk] = []

        for path in path_list:
            text: str = self.read_file(path)
            suffix: str = path.suffix

            list_chunks.extend(
                cutchunk.chunk_file(str(path), text, max_chunk_size, suffix)
            )
        return list_chunks
