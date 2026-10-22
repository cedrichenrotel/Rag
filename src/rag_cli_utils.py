from pathlib import Path

import ollama

from src.bm25_index import Bm25Index
from src.corpus import ChunkCorpus
from src.model import Chunk, MinimalSource
from src.ollama_generator import OllamaGenerator
from src.path import PathFile
from src.qwen_generator import QwenGenerator


class RagCliUtils:
    def __init__(self) -> None:
        self._chunkcorpus: ChunkCorpus = ChunkCorpus()
        self._pathfile: PathFile = PathFile()

    def _check_query(self, query: str) -> bool:
        """verifie le bon typage et les bonne valeur de query
        creer par l'utilisateur

        Args:
            query (str): question posee

        Returns:
            bool: si le typage et la veleur son bonne retourne True
        """

        is_valid: bool = True
        if not isinstance(query, str) or query.strip() == "":
            print("[ERROR] query must be a non-empty string.")
            is_valid = False
        return is_valid

    def _check_k(self, k: int) -> bool:
        """verifie le bon typage et les bonne valeur des k
        creer par l'utilisateur

        Args:
            k (int): variable donne le nombre de resultat a sortir

        Returns:
            bool: si le typage et la veleur son bonne retourne True
        """

        is_valid: bool = True
        if not isinstance(k, int) or isinstance(k, bool) or k < 1:
            print(f"[ERROR] k must be an integer >= 1 (got {k})")
            is_valid = False
        return is_valid

    def _choose_generator(self) -> QwenGenerator:
        """verifie si le serveur Ollama est bien
        fonctionnel, si non envois un message pour
        avertir que le gnerator utlisera qwen par
        defaut

        Returns:
            QwenGenerator: Ollamgenerator | generator (par defaut)
        """

        try:
            ollama.list()
        except ConnectionError as e:
            print(f"[WARNING] Choose_generator: {e} -> activate Qwen default")
            return QwenGenerator()
        return OllamaGenerator()

    def _read_source(self, source: MinimalSource) -> str:
        """elle retourne un passage du texte a partir de son
        adresse(fichier + index).

        Args:
            source (MinimalSource): reponse generer par bm25

        Returns:
            str: retourne le passage entre les 2 index
        """

        text: str = self._chunkcorpus.read_file(Path(source.file_path))
        return text[source.first_character_index:source.last_character_index]

    def _print_result(
        self, query: str, top_chunks: list[Chunk], answer: str | None = None
    ) -> None:
        """affiche les resultats generer par les fonction 'def search()' ou
        'def answer().

        Args:
            query (str): question a repondre
            top_chunks (list[Chunk]): resultat des meilleur chunk generer par
            bm25.
            answer (str | None, optional): reponse generer par le model de
            language. Defaults to None.
        """

        print(f"\nQuery: {query}")
        print("\n        __ RANKINGS __  \n")
        for i, chunk in enumerate(top_chunks, start=1):
            print(f"Top {i}: {chunk.file_path}")
            print(
                f"    - First_character_index: {chunk.first_character_index}"
            )
            print(f"    - Last_character_index: {chunk.last_character_index}")
        if answer:
            print("\n     __ GENERATE ANSWER __\n")
            print(f"    - Answer: {answer}")
        print()

    def _to_sources(self, chunks: list[Chunk]) -> list[MinimalSource]:
        """Convertis une liste de chunk en une liste
        list[MinimalSource]

        Args:
            chunks (list[chunk]): liste d'indexation que bm25 renvois

        Returns:
            list[MinimalSource]: retourne une liste de classe MinimalSource
        """

        sources: list[MinimalSource] = []
        for chunk in chunks:
            sources.append(
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
            )
        return sources

    def _load_bm25(self, path: str) -> Bm25Index:
        """Charge les chunks du fichier d'index et construit l'outil
        de recherche BM25.
        arg:
            - index_path -> chemin du fichier d'index
        return: le Bm25Index pret a recevoir des questions"""

        list_chunk: list[Chunk] = self._pathfile.load(Path(path))
        bm25: Bm25Index = Bm25Index(list_chunk)
        return bm25
