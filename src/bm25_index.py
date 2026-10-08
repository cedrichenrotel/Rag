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
    cutter = CutChunk()
    text = (
        "def load_lora():\n    pass\n\n"
        "def save_model():\n    pass\n\n"
        "class ChunkCutter:\n    pass\n"
    )
    list_chunk = cutter.chunk_file("test.py", text, 2000, ".py")
    bm = Bm25Index(list_chunk)
    query = "How to load LoRA?"
    query_token = bm.tokenizer.tokenize(query)
    print(len(bm.list_text), f"text: {bm.list_text}")
    print()
    print(f"tokenize_text: {bm.tokenize_text()}")
    print(f"query_token: {query_token}")
    print(f"bm25: {bm.bm25.get_scores(query_token)}")
