import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Verifie qu'il y a bien que ses 3 attribut dans
    dans 'sources':list[dict[str, Any]]. Cette classe
    permet de connaitre le chemin du fichier, ou commence
    et termine les caracters de la reponse"""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """verife qu'il y est bien une question sans reponse.
    L'identifiant est generer automatiquement si'il est absent"""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Une question du corrige : la question, ses sources et la reponse
    attendue"""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """permet d'utiliser une seul class les 4 fichier data/dataset/ au
    lieu d'un par fichier"""
    rag_questions: list[AnsweredQuestion | UnansweredQuestion]
