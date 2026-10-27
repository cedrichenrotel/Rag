from pathlib import Path

import tqdm

from src.chunk import CutChunk
from src.model import Chunk


class ChunkCorpus:
    """Read the corpus and split it into chunks.

    Collect the useful files (.md, .txt, .py), then call
    CutChunk to split each text according to its file type.
    """

    def list_files(self, raw_path: Path) -> list[Path]:
        """Walk through all the files of the data/raw folder and build
        a list of paths of every file whose suffix is
        .py, .md or .txt.
        args:
            - raw_path: raw data folder"""

        if not raw_path.is_dir():
            raise ValueError(
                f"[WARNING] list_files: Not a directory: {raw_path}"
            )

        list_file: list[Path] = []
        for path in raw_path.rglob("*"):
            if path.is_file() and path.suffix in (".md", ".py", ".txt"):
                list_file.append(path)

        return list_file

    def read_file(self, file: Path) -> str:
        """Return the text of the file at the given path."""

        return file.read_text(encoding="utf-8")

    def chunk_corpus(
        self, raw_path: Path, max_chunk_size: int, overlap: int = 0
    ) -> list[Chunk]:
        """Return a list of chunks for the whole corpus.
        args:
            - raw_path: raw data folder
            - max_chunk_size: maximum size of a chunk
            - overlap: overlap percentage between 2 chunks"""

        path_list: list[Path] = self.list_files(raw_path)
        if not path_list:
            raise FileNotFoundError(
                "[WARNING] chunk_corpus: File (“.md”/.txt'/“.py”) not found"
            )

        cutchunk: CutChunk = CutChunk()
        list_chunks: list[Chunk] = []

        for path in tqdm.tqdm(path_list, desc="Indexing"):
            text: str = self.read_file(path)
            suffix: str = path.suffix

            list_chunks.extend(
                cutchunk.chunk_file(
                    str(path), text, max_chunk_size, suffix, overlap
                )
            )
        return list_chunks
