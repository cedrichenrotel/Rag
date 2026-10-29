import json
from pathlib import Path
from typing import Any

from src.model import (
    Chunk,
    RagDataset,
    StudentSearchResults,
    StudentSearchResultsAndAnswer,
)


class PathFile:
    def create_json_missing(self, file: Path) -> None:
        """Create all the missing parent folders (e.g. data/processed).
        arg:
            - file -> target file"""

        file.parent.mkdir(parents=True, exist_ok=True)

    def save(self, chunks: list[Chunk], path: Path) -> None:
        """Convert the chunks into dicts and save them as JSON.
        args:
            - path -> path of the file to create
            (e.g. data/processed/index.json).
            If it already exists, it is replaced.
            - chunks -> list of chunks with all the information needed
            to store them in the JSON file (used by index)."""

        self.create_json_missing(path)

        with open(path, "w") as f:
            data: list[dict[str, Any]] = [
                chunk.model_dump() for chunk in chunks
            ]
            json.dump(data, f, indent=4)

    def load(self, path: Path) -> list[Chunk]:
        """Get back the list of chunks saved by save, without reading
        or splitting the corpus again (used by search).
        arg:
            - path -> target file"""

        with open(path, "r") as f:
            data: list[dict[str, Any]] = json.load(f)
            list_chunks: list[Chunk] = [Chunk.model_validate(d) for d in data]
        return list_chunks

    def load_dataset(self, path: Path) -> RagDataset:
        """Open a dataset file (e.g. dataset_docs_public.json) and get
        a RagDataset object.
        args:
            - path -> target file"""

        with open(path, "r") as f:
            data: list[dict[str, Any]] = json.load(f)
            dataset: RagDataset = RagDataset.model_validate(data)
        return dataset

    def load_results(self, path: Path) -> StudentSearchResults:
        """Read the file produced by search_dataset and get a
        StudentSearchResults object (used by evaluate).
        args:
            - path -> target file"""

        with open(path, "r") as f:
            data: dict[str, Any] = json.load(f)
            results: StudentSearchResults = (
                StudentSearchResults.model_validate(data)
            )
        return results

    def save_result(
        self,
        student_results: StudentSearchResults | StudentSearchResultsAndAnswer,
        output_path: Path,
    ) -> None:
        """Create a JSON file (e.g. dataset_docs_public.json) and store
        the results in it.
        args:
            - student_results -> results of search_dataset or
            answer_dataset
            - output_path -> e.g. data/output/search_results/
            UnansweredQuestions/dataset_docs_public.json"""

        self.create_json_missing(output_path)
        with open(output_path, "w") as f:
            data: dict[str, Any] = student_results.model_dump()
            json.dump(data, f, indent=4)
