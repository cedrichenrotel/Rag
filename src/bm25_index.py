from rank_bm25 import BM25Okapi

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

    def list_scores(self, query: str) -> list[tuple[float, Chunk]]:
        """Retourne les paires (note, chunk) triees de la meilleure
        note a la moins bonne pour la question donnee.

        query: question permettra a bm25 de donner une note au chunk"""

        query_token: list[str] = self.tokenizer.tokenize(query)
        note: list[float] = self.bm25.get_scores(query_token)
        list_scores: list[tuple[float, Chunk]] = sorted(
            zip(note, self.list_chunks),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return list_scores

    def search(self, query: str, k: int) -> list[Chunk]:
        """retourne  les k meilleurs chunk de la liste des scores

        query: question
        k: nombre des milleurs chunk a retourner"""

        list_scores: list[tuple[float, Chunk]] = self.list_scores(query)
        return [chunk for _, chunk in list_scores[:k]]
