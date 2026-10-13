from pathlib import Path

from src.bm25_index import Bm25Index
from src.corpus import ChunkCorpus
from src.model import (
    Chunk,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)
from src.path import PathFile


class RagCli:
    """class contenant toutes les methodes pour
    exectuter les commande d'entree que fire
    transformera"""

    def __init__(self) -> None:
        self.chunkcorpus: ChunkCorpus = ChunkCorpus()
        self.pathfile: PathFile = PathFile()

    def _load_bm25(self, path: str) -> Bm25Index:
        """Charge les chunks du fichier d'index et construit l'outil
        de recherche BM25.
        arg:
            - index_path -> chemin du fichier d'index
        return: le Bm25Index pret a recevoir des questions"""

        list_chunk: list[Chunk] = self.pathfile.load(Path(path))
        bm25: Bm25Index = Bm25Index(list_chunk)
        return bm25

    def index(
        self,
        max_chunk_size: int = 2000,
        raw_path: str = "data/raw",
        index_path: str = "data/processed/index.json",
    ) -> None:
        """Decoupe les fichiers en paquet de 2000 caracteres
        max dans data/raw/vllm-0.10.1 et sauvegarde l'index
        dans data/processed
        args:
            - max_chunk_size -> taille max chunk
            - raw_path -> donnee brut
            - index_path -> fichier stock resultat index"""

        list_chunk: list[Chunk] = self.chunkcorpus.chunk_corpus(
            Path(raw_path), max_chunk_size
        )
        self.pathfile.save(list_chunk, Path(index_path))

    def search(
        self,
        query: str,
        k: int = 10,
        index_path: str = "data/processed/index.json",
    ) -> None:
        """permet de retourner les k meilleurs chunk du
        classement d'une question.
            args:
                - query -> questions posee
                - k -> nombre de chunk"""

        bm25: Bm25Index = self._load_bm25(index_path)
        top_chunks: list[Chunk] = bm25.search(query, k)
        print(f"\nQuery: {query}")
        print("\n        __ Rankings __  \n")
        for i, chunk in enumerate(top_chunks, start=1):
            print(f"Top {i}: {chunk.file_path}")
            print(
                f"    - First_character_index: {chunk.first_character_index}"
            )
            print(f"    - Last_character_index: {chunk.last_character_index}")
            print()

    def search_dataset(
        self,
        dataset_path: str,
        save_directory: str,
        k: int = 10,
        index_path: str = "data/processed/index.json",
    ) -> None:
        """retourne les k meilleurs chunk d'un fichier de question
        et l'enregistre dans fichier JSON.
            args:
                - dataset_path -> chemin du fichier JSON contenant
                  les questions
                - save_directory -> dossier ou est stocker le fichier
                  resultat de recherche
                - k -> nombre de chunk a retourner par question"""

        dataset: RagDataset = self.pathfile.load_dataset(Path(dataset_path))
        bm25: Bm25Index = self._load_bm25(index_path)
        list_results: list[MinimalSearchResults] = []
        for question in dataset.rag_questions:
            top_chunks: list[Chunk] = bm25.search(question.question, k)
            sources: list[MinimalSource] = [
                MinimalSource(
                    file_path=chunk.file_path,
                    first_character_index=chunk.first_character_index,
                    last_character_index=chunk.last_character_index,
                )
                for chunk in top_chunks
            ]
            result: MinimalSearchResults = MinimalSearchResults(
                question_id=question.question_id,
                question=question.question,
                retrieved_sources=sources,
            )
            list_results.append(result)
        student_result: StudentSearchResults = StudentSearchResults(
            search_results=list_results, k=k
        )
        output_path: Path = Path(save_directory) / Path(dataset_path).name
        self.pathfile.save_result(student_result, output_path)

    def answer(self, query: str, k: int = 10):
        """retourne les k meilleurs chunk d'une question et l'envois
        au model Qwen afin de rediger une reponse en texte.
            args:
                - query -> question posee
                - k -> nombre de chunk a retourner"""

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ):
        """redige les reponses de toutes les questions d'un fichier et les
        enregistre dans un fichier JSON (elle lis le fichier produit par
        search_dataset).
            args:
                - student_search_results_path -> chemin du fichier JSON
                  produit par la fonction search_dataset
                - save_directory -> dossier ou est stocker le fichier des
                  reponses"""

    def evaluate(self, student_search_results_path: str, dataset_path: str):
        """compare les resultats de recherche avec le corriger et creer un
        pourcentage de reussite.
            args:
                - student_search_results_path -> chemin du fichier JSON
                  produit par la fonction search_dataset
                - dataset_path -> chemin du fichier JSON corrige
                (AnsweredQuestions))"""
