from rank_bm25 import BM25Okapi

from src.model import Chunk
from src.tokenizer import Tokenizer


class Bm25Index:
    """Give a score to each chunk
    and keep the best-scored ones."""

    def __init__(self, list_chunks: list[Chunk]) -> None:
        self.list_chunks: list[Chunk] = list_chunks
        self.tokenizer: Tokenizer = Tokenizer()
        self.bm25: BM25Okapi = BM25Okapi(self._tokenize_text(), k1=1)

    def _tokenize_text(self) -> list[list[str]]:
        """Convert chunk.text and chunk.file_path into a list of tokens."""

        list_tokens: list[list[str]] = []
        for chunk in self.list_chunks:
            tokens: list[str] = self.tokenizer.tokenize(chunk.text)
            tokens.extend(self.tokenizer.tokenize(chunk.file_path))
            list_tokens.append(tokens)
        return list_tokens

    def _list_scores(self, query: str) -> list[tuple[float, Chunk]]:
        """Return the (score, chunk) pairs sorted from the best
        score to the worst for the given question.

        query: question used by BM25 to give a score to each chunk"""

        query_token: list[str] = self.tokenizer.tokenize(query)
        note: list[float] = self.bm25.get_scores(query_token)
        list_scores: list[tuple[float, Chunk]] = sorted(
            zip(note, self.list_chunks),
            key=lambda pair: pair[0],
            reverse=True,
        )
        return list_scores

    def search(self, query: str, k: int) -> list[Chunk]:
        """Return the k best chunks from the list of scores.

        query: question
        k: number of best chunks to return"""

        list_scores: list[tuple[float, Chunk]] = self._list_scores(query)
        return [chunk for _, chunk in list_scores[:k]]
