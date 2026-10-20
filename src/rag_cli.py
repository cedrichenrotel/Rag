from pathlib import Path

import tqdm

from src.bm25_index import Bm25Index
from src.corpus import ChunkCorpus
from src.evaluate import Evaluator
from src.model import (
    Chunk,
    MinimalAnswer,
    MinimalSearchResults,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)
from src.ollama_generator import OllamaGenerator
from src.path import PathFile
from src.qwen_generator import QwenGenerator


class RagCli:
    """class contenant toutes les methodes pour
    exectuter les commande d'entree que fire
    transformera"""

    def __init__(self) -> None:
        self.chunkcorpus: ChunkCorpus = ChunkCorpus()
        self.pathfile: PathFile = PathFile()
        self.evaluator: Evaluator = Evaluator()

    def _read_source(self, source: MinimalSource) -> str:
        """elle retourne un passage du texte a partir de son
        adresse(fichier + index).

        Args:
            source (MinimalSource): reponse generer par bm25

        Returns:
            str: retourne le passage entre les 2 index
        """

        text: str = self.chunkcorpus.read_file(Path(source.file_path))
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
                - k -> nombre de chunk
                - index_path -> chemin du fichier indexer"""

        bm25: Bm25Index = self._load_bm25(index_path)
        top_chunks: list[Chunk] = bm25.search(query, k)
        self._print_result(query, top_chunks, None)

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
            sources: list[MinimalSource] = self._to_sources(top_chunks)
            result: MinimalSearchResults = MinimalSearchResults(
                question_id=str(question.question_id),
                question=question.question,
                retrieved_sources=sources,
            )
            list_results.append(result)
        student_result: StudentSearchResults = StudentSearchResults(
            search_results=list_results, k=k
        )
        output_path: Path = Path(save_directory) / Path(dataset_path).name
        self.pathfile.save_result(student_result, output_path)

    def answer(
        self,
        query: str,
        k: int = 10,
        index_path: str = "data/processed/index.json",
        max_new_tokens: int = 100,
    ) -> None:
        """Affiche les k meilleurs chunk d'une question et l'envois
        au model Qwen afin de rediger une reponse en texte.
            args:
                - query -> question posee
                - k -> nombre de chunk a retourner
                - index_path -> chemin fichier sont stcker les chunks pour
                BM25"""

        generator: QwenGenerator = QwenGenerator()
        bm25: Bm25Index = self._load_bm25(index_path)
        top_chunks: list[Chunk] = bm25.search(query, k)

        list_texts: list[str] = [chunk.text for chunk in top_chunks]
        answer: str = generator.generate_answer(
            query, list_texts, max_new_tokens
        )
        self._print_result(query, top_chunks, answer)

    def answer_dataset(
        self,
        student_search_results_path: str,
        save_directory: str,
        max_new_tokens: int = 100,
        max_context_tokens: int = 1000,
    ) -> None:
        """redige les reponses de toutes les questions d'un fichier et les
        enregistre dans un fichier JSON (elle lis le fichier produit par
        search_dataset).
            args:
                - student_search_results_path -> chemin du fichier JSON
                  produit par la fonction search_dataset
                - save_directory -> dossier ou est stocker le fichier des
                  reponses
                - max_new_tokens -> prend les 100 meilleurs token pour raccourcir
                le delais d'attente
                - """

        student: StudentSearchResults = self.pathfile.load_results(
            Path(student_search_results_path)
        )
        generator: OllamaGenerator = OllamaGenerator()

        list_minimalanswer: list[MinimalAnswer] = []
        for result in tqdm.tqdm(student.search_results, desc="Answering"):
            list_text: list[str] = []
            for source in result.retrieved_sources:
                text: str = self._read_source(source)
                list_text.append(text)
            answer: str = generator.generate_answer(
                result.question, list_text, max_new_tokens, max_context_tokens
            )
            list_minimalanswer.append(
                MinimalAnswer(
                    question_id=result.question_id,
                    question=result.question,
                    retrieved_sources=result.retrieved_sources,
                    answer=answer,
                )
            )

        student_result: StudentSearchResultsAndAnswer = (
            StudentSearchResultsAndAnswer(
                search_results=list_minimalanswer, k=student.k
            )
        )
        output_path: Path = (
            Path(save_directory) / Path(student_search_results_path).name
        )
        self.pathfile.save_result(student_result, output_path)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> None:
        """compare les resultats de recherche avec le corriger et creer un
        pourcentage de reussite.
            args:
                - student_search_results_path -> chemin du fichier JSON
                  produit par la fonction search_dataset
                - dataset_path -> chemin du fichier JSON corrige
                (AnsweredQuestions))
        retour: une moyenne % de bonne reponse sur la liste de toute les
        questions"""

        dataset: RagDataset = self.pathfile.load_dataset(Path(dataset_path))
        student: StudentSearchResults = self.pathfile.load_results(
            Path(student_search_results_path)
        )
        score: float = self.evaluator.recall_dataset(dataset, student, k)
        print(f"recall@: {score:.1%}")
