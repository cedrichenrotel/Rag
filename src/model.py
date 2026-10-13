import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Localise un passage dans un fichier du corpus.
    Contient:
        - file_path -> chemin du fichier d'ou vient le passage
        - first_character_index -> index du debut du passage
        - last_character_index -> index de fin du passage"""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Question sans reponse.
    Contient:
        - question_id -> son identifiant
        (genere automatiquement s'il est absent)
        - question -> le texte de la question"""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Question avec son corrige (herite de UnansweredQuestion).
    Contient:
        - sources -> liste des passages ou se trouve la reponse
        - answer -> reponse attendue"""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Contenu d'un fichier de dataset.
    Contient:
        - rag_questions -> liste des questions du fichier,
        avec ou sans corrige"""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class Chunk(MinimalSource):
    """Morceau d'un fichier du corpus (herite de MinimalSource).
    Contient:
        - text -> texte compris entre les deux index du passage"""

    text: str


class MinimalSearchResults(BaseModel):
    """Resultat de recherche d'une question.
    Contient:
        - question_id -> identifiant de la question
        - question -> le texte de la question
        - retrieved_sources -> liste des passages trouves
        (fichier, index debut et fin)"""

    question_id: str
    question: str
    retrieved_sources: list[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """Resultat de recherche d'une question, avec sa reponse
    (herite de MinimalSearchResults).
    Contient:
        - answer -> reponse redigee par Qwen (modele de langage)"""

    answer: str


class StudentSearchResults(BaseModel):
    """Contenu du fichier produit par search_dataset.
    Contient:
        - search_results -> liste de MinimalSearchResults,
        un par question
        - k -> nombre de passages gardes pour chaque question"""

    search_results: list[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Contenu du fichier produit par answer_dataset.
    Contient:
        - search_results -> liste de MinimalAnswer, un par question
        - k -> nombre de passages gardes pour chaque question"""

    search_results: list[MinimalAnswer]
    k: int
