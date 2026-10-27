import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Locate a passage in a file of the corpus.
    Contains:
        - file_path -> path of the file the passage comes from
        - first_character_index -> index of the start of the passage
        - last_character_index -> index of the end of the passage"""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Question without an answer.
    Contains:
        - question_id -> its identifier
        (generated automatically if missing)
        - question -> the text of the question"""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question with its answer key (inherits from UnansweredQuestion).
    Contains:
        - sources -> list of passages where the answer is found
        - answer -> expected answer"""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Content of a dataset file.
    Contains:
        - rag_questions -> list of the questions of the file,
        with or without an answer key"""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class Chunk(MinimalSource):
    """Piece of a file of the corpus (inherits from MinimalSource).
    Contains:
        - text -> text between the two indexes of the passage"""

    text: str


class MinimalSearchResults(BaseModel):
    """Search result for one question.
    Contains:
        - question_id -> identifier of the question
        - question -> the text of the question
        - retrieved_sources -> list of the passages found
        (file, start and end index)"""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Search result for one question, with its answer
    (inherits from MinimalSearchResults).
    Contains:
        - answer -> answer written by Qwen (language model)"""

    answer: str


class StudentSearchResults(BaseModel):
    """Content of the file produced by search_dataset.
    Contains:
        - search_results -> list of MinimalSearchResults,
        one per question
        - k -> number of passages kept for each question"""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Content of the file produced by answer_dataset.
    Contains:
        - search_results -> list of MinimalAnswer, one per question
        - k -> number of passages kept for each question"""

    search_results: list[MinimalAnswer]
    k: int
