from rank_bm25 import BM25Okapi

from src.chunk import CutChunk
from src.model import Chunk
from src.tokenizer import Tokenizer


class Bm25Index:
    """Permet de donner une note à chaque chunk
    et garder le mieux noteés"""

    def __init__(self, list_chunks: list[Chunk]) -> None:
        self.list_chunks: list[Chunk] = list_chunks
        self.tokenizer: Tokenizer = Tokenizer()
        self.list_text: list[str] = [chunk.text for chunk in list_chunks]
        self.bm25: BM25Okapi = BM25Okapi(self.tokenize_text())

    def tokenize_text(self) -> list[list[str]]:
        """Convertir le chunks.texte en une liste de token"""

        list_tokens: list[list[str]] = []
        for text in self.list_text:
            list_tokens.append(self.tokenizer.tokenize(text))
        return list_tokens


if __name__ == "__main__":
    chunk = CutChunk()
    chunks: list[Chunk] = chunk.chunk_file(
        "src/chunk.py", "def test\ndefdef|\n    class trololo" * 2, 10, ".py"
    )
    bm = Bm25Index(chunks)
    print(len(bm.list_text), f"text: {bm.list_text}")
    print()
    print(f"tokenize_text: {bm.tokenize_text()}")
    print(f"bm25: {bm.bm25}")
