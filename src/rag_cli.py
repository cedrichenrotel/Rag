from pathlib import Path

import tqdm

from src.bm25_index import Bm25Index
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
from src.qwen_generator import QwenGenerator
from src.rag_cli_utils import RagCliUtils


class RagCli(RagCliUtils):
    """class contenant toutes les methodes pour
    exectuter les commande d'entree que fire
    transformera"""

    def __init__(self) -> None:
        super().__init__()
        self._evaluator: Evaluator = Evaluator()

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

        if (
            not isinstance(max_chunk_size, int)
            or isinstance(max_chunk_size, bool)
            or not 1 <= max_chunk_size <= 2000
        ):
            print(
                "[ERROR] index: max_chunk_size must be an int "
                "between 1 and 2000"
            )
            return

        try:
            list_chunk: list[Chunk] = self._chunkcorpus.chunk_corpus(
                Path(raw_path), max_chunk_size
            )
            self._pathfile.save(list_chunk, Path(index_path))
        except (ValueError, OSError) as e:
            print(f"[ERROR] Index: {e}")
            return
        print(f"{len(list_chunk)} chunks saved in {index_path}")

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

        if not self._check_query(query) or not self._check_k(k):
            return

        try:
            bm25: Bm25Index = self._load_bm25(index_path)
        except (ValueError, OSError) as e:
            print(f"[ERROR] search: {e}")
            return
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

        if not self._check_k(k):
            return
        try:
            dataset: RagDataset = self._pathfile.load_dataset(
                Path(dataset_path)
            )
            bm25: Bm25Index = self._load_bm25(index_path)
        except (ValueError, OSError) as e:
            print(
                f"[ERROR] search_dataset: {dataset_path}|{index_path} -> {e}"
            )
            return
        list_results: list[MinimalSearchResults] = []
        for question in tqdm.tqdm(dataset.rag_questions, desc="Searching"):
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
        try:
            self._pathfile.save_result(student_result, output_path)
        except (ValueError, OSError) as e:
            print(f"[ERROR] search_dataset: {save_directory} -> {e}")
            return

    def answer(
        self,
        query: str,
        k: int = 10,
        index_path: str = "data/processed/index.json",
        max_new_tokens: int = 100,
        max_context_tokens: int = 1000,
    ) -> None:
        """Affiche les k meilleurs chunk d'une question et l'envois
        au model Qwen afin de rediger une reponse en texte.
            args:
                - query -> question posee
                - k -> nombre de chunk a retourner
                - index_path -> chemin fichier sont stcker les chunks pour
                  BM25
                - max_new_tokens -> Permet de recouper la longhueur maximale
                  de la reponse ecrite par qwen
                - max_context_token -> les meilleurs chunks tant que le total
                  reste sous ce nombre de tokens"""

        if not self._check_token(
            "max_new_tokens", max_new_tokens
        ) or not self._check_token("max_context_tokens", max_context_tokens):
            return

        if not self._check_query(query) or not self._check_k(k):
            return

        try:
            bm25: Bm25Index = self._load_bm25(index_path)
        except (ValueError, OSError) as e:
            print(f"[ERROR] answer: {e}")
            return
        generator: QwenGenerator = QwenGenerator()
        top_chunks: list[Chunk] = bm25.search(query, k)

        list_texts: list[str] = [chunk.text for chunk in top_chunks]
        answer: str = generator.generate_answer(
            query, list_texts, max_new_tokens, max_context_tokens
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
                - max_new_tokens -> Permet de recouper la longhueur maximale
                  de la reponse ecrite par qwen
                - max_context_tokens -> les meilleurs chunks tant que le total
                  reste sous ce nombre de tokens"""

        if not self._check_token(
            "max_new_tokens", max_new_tokens
        ) or not self._check_token("max_context_tokens", max_context_tokens):
            return

        try:
            student: StudentSearchResults = self._pathfile.load_results(
                Path(student_search_results_path)
            )
        except (ValueError, OSError) as e:
            print(f"[ERROR] answer_dataset: {e}")
            return

        generator: QwenGenerator = self._choose_generator()

        list_minimalanswer: list[MinimalAnswer] = []
        try:
            for result in tqdm.tqdm(student.search_results, desc="Answering"):
                list_text: list[str] = []
                for source in result.retrieved_sources:
                    try:
                        text: str = self._read_source(source)
                    except (ValueError, OSError) as e:
                        print(f"[WARNING] answer_dataset: {e}")
                        continue
                    list_text.append(text)
                answer: str = generator.generate_answer(
                    result.question,
                    list_text,
                    max_new_tokens,
                    max_context_tokens,
                )
                list_minimalanswer.append(
                    MinimalAnswer(
                        question_id=result.question_id,
                        question=result.question,
                        retrieved_sources=result.retrieved_sources,
                        answer=answer,
                    )
                )
        except KeyboardInterrupt:
            print(
                "[WARNING] answer_dataset: stopped, saving "
                f"{len(list_minimalanswer)} answers"
            )

        student_result: StudentSearchResultsAndAnswer = (
            StudentSearchResultsAndAnswer(
                search_results=list_minimalanswer, k=student.k
            )
        )
        output_path: Path = (
            Path(save_directory) / Path(student_search_results_path).name
        )
        try:
            self._pathfile.save_result(student_result, output_path)
        except (ValueError, OSError) as e:
            print(f"[ERROR] answer_dataset: {save_directory} -> {e}")
            return

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

        if not self._check_k(k):
            return
        try:
            dataset: RagDataset = self._pathfile.load_dataset(
                Path(dataset_path)
            )
            student: StudentSearchResults = self._pathfile.load_results(
                Path(student_search_results_path)
            )
        except (ValueError, OSError) as e:
            print(f"[ERROR] evaluate: {e}")
            return
        if not dataset.rag_questions:
            print(f"[ERROR] evaluate: dataset see: {dataset_path}")
            return
        score: float = self._evaluator.recall_dataset(dataset, student, k)
        print(f"recall@{k}: {score:.1%}")
