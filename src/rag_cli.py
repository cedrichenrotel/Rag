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

DEFAULT_CHUNK_SIZE = 2000
DEFAULT_OVERLAP = 20


class RagCli(RagCliUtils):
    """Class containing all the methods that Fire
    turns into CLI commands."""

    def __init__(self) -> None:
        super().__init__()
        self._evaluator: Evaluator = Evaluator()

    def index(
        self,
        max_chunk_size: int = DEFAULT_CHUNK_SIZE,
        raw_path: str = "data/raw",
        index_path: str = "data/processed/index.json",
        overlap: int = DEFAULT_OVERLAP,
    ) -> None:
        """Split the files of data/raw/vllm-0.10.1 into chunks of
        at most 2000 characters and save the index
        in data/processed.
        args:
            - max_chunk_size -> maximum size of a chunk
            - raw_path -> raw data folder
            - index_path -> file where the index is stored
            - overlap -> overlap percentage (0-99) between 2 chunks"""

        if (
            not isinstance(max_chunk_size, int)
            or isinstance(max_chunk_size, bool)
            or not 100 <= max_chunk_size <= 2000
        ):
            print(
                "[WARNING] index: Incorrect input value, max_chunk_size must"
                " be an int between 100 and 2000."
            )
            max_chunk_size = DEFAULT_CHUNK_SIZE
            print(f"Value default: {DEFAULT_CHUNK_SIZE}")

        if (
            not isinstance(overlap, int)
            or isinstance(overlap, bool)
            or not 0 <= overlap <= 99
        ):
            print(
                f"[WARNING] index: Incorrect input value: {overlap},"
                " overlap must be an int between 0 and 99"
            )
            overlap = DEFAULT_OVERLAP
            print(f"Value default: {DEFAULT_OVERLAP}")

        try:
            list_chunk: list[Chunk] = self._chunkcorpus.chunk_corpus(
                Path(raw_path), max_chunk_size, overlap
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
        """Display the k best-ranked chunks for a question.
        args:
            - query -> question asked
            - k -> number of chunks
            - index_path -> path of the index file"""

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
        """Find the k best chunks for every question of a file
        and save them in a JSON file.
            args:
                - dataset_path -> path of the JSON file containing
                  the questions
                - save_directory -> folder where the search results
                  file is stored
                - k -> number of chunks to return per question"""

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
        print()
        print(
            f"{len(student_result.search_results)} results saved in"
            f" {output_path}\n"
        )

    def answer(
        self,
        query: str,
        k: int = 10,
        index_path: str = "data/processed/index.json",
        max_new_tokens: int = 100,
        max_context_tokens: int = 1000,
    ) -> None:
        """Display the k best chunks for a question and send them
        to the Qwen model to write a text answer.
            args:
                - query -> question asked
                - k -> number of chunks to return
                - index_path -> path of the file where the chunks for
                  BM25 are stored
                - max_new_tokens -> maximum length of the answer
                  written by Qwen
                - max_context_tokens -> keep the best chunks as long as the
                  total stays under this number of tokens"""

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
        """Write the answers to all the questions of a file and
        save them in a JSON file (it reads the file produced by
        search_dataset).
            args:
                - student_search_results_path -> path of the JSON file
                  produced by search_dataset
                - save_directory -> folder where the answers file
                  is stored
                - max_new_tokens -> maximum length of the answer
                  written by Qwen
                - max_context_tokens -> keep the best chunks as long as the
                  total stays under this number of tokens"""

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
        print()
        print(
            f"{len(student_result.search_results)} answers saved in"
            f" {output_path}\n"
        )

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
        k: int = 10,
    ) -> None:
        """Compare the search results with the answer key and compute
        a success rate.
            args:
                - student_search_results_path -> path of the JSON file
                  produced by search_dataset
                - dataset_path -> path of the answer key JSON file
                (AnsweredQuestions)
        returns: the average % of correct answers over all
        the questions"""

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
