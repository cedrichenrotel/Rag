import os

from rank_bm25 import BM25Okapi

from src.model import Chunk
from src.tokenizer import Tokenizer

# Dossiers annexes (tests, exemples...) : beaucoup de mots en commun avec
# les questions, mais ce ne sont pas les vraies sources. Leur note est reduite.
SECONDARY_DIRS: tuple[str, ...] = (
    "tests",
    "examples",
    "benchmarks",
    ".buildkite",
    "tools",
)
SECONDARY_PENALTY: float = 0.5


class Bm25Index:
    """Permet de donner une note à chaque chunk
    et garder le mieux noteés"""

    def __init__(self, list_chunks: list[Chunk]) -> None:
        self.list_chunks: list[Chunk] = list_chunks
        self.tokenizer: Tokenizer = Tokenizer()
        self.bm25: BM25Okapi = BM25Okapi(self._tokenize_text(), k1=1)
        self.penalties: list[float] = self._compute_penalties()

    def _tokenize_text(self) -> list[list[str]]:
        """Convertir le chunks.texte et chunks.path en une liste de token"""

        list_tokens: list[list[str]] = []
        for chunk in self.list_chunks:
            tokens: list[str] = self.tokenizer.tokenize(chunk.text)
            tokens.extend(self.tokenizer.tokenize(chunk.file_path))
            list_tokens.append(tokens)
        return list_tokens

    def _compute_penalties(self) -> list[float]:
        """Retourne, pour chaque chunk, le coefficient applique a sa note :
        SECONDARY_PENALTY si son premier dossier sous la racine du corpus
        est dans SECONDARY_DIRS, sinon 1. La racine est le debut de chemin
        commun a tous les chunks (ex: data/raw/vllm-0.10.1)."""

        if not self.list_chunks:
            return []
        root: str = os.path.commonpath(
            [os.path.dirname(chunk.file_path) for chunk in self.list_chunks]
        )
        penalties: list[float] = []
        for chunk in self.list_chunks:
            relative: str = os.path.relpath(chunk.file_path, root)
            if relative.split(os.sep)[0] in SECONDARY_DIRS:
                penalties.append(SECONDARY_PENALTY)
            else:
                penalties.append(1.0)
        return penalties

    def _list_scores(self, query: str) -> list[tuple[float, Chunk]]:
        """Retourne les paires (note, chunk) triees de la meilleure
        note a la moins bonne pour la question donnee.

        query: question permettra a bm25 de donner une note au chunk"""

        query_token: list[str] = self.tokenizer.tokenize(query)
        note: list[float] = self.bm25.get_scores(query_token)
        note = [score * pen for score, pen in zip(note, self.penalties)]
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

        list_scores: list[tuple[float, Chunk]] = self._list_scores(query)
        return [chunk for _, chunk in list_scores[:k]]
