import uuid

from pydantic import BaseModel, Field


class MinimalSource(BaseModel):
    """Localise un passage dans un fichier du corpus.
    file_path est le chemin du fichier d'ou vient le passage.
    first_character_index et last_character_index sont les
    positions du premier et du dernier caractere du passage.
    Les trois champs sont obligatoires : pydantic refuse la
    donnee si l'un d'eux manque ou n'a pas le bon type."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """Decrit une question sans reponse.
    question est le texte de la question, il est obligatoire.
    question_id est son identifiant : s'il est absent des
    donnees, un nouvel identifiant unique (uuid) est genere
    automatiquement."""

    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """Decrit une question via sont heritage, sa source et sa reponse"""

    sources: list[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """Decrit le contenu d'un fichier de dataset.
    rag_questions est la liste des questions du fichier. Chaque
    question est soit une AnsweredQuestion (avec sources et
    reponse), soit une UnansweredQuestion (question seule).
    Une seule classe suffit donc pour lire les 4 fichiers de
    datasets, au lieu d'une par type de fichier."""

    rag_questions: list[AnsweredQuestion | UnansweredQuestion]


class Chunk(MinimalSource):
    """Morceau d'un fichier du corpus, produit par le decoupage.
    Herite de MinimalSource : file_path est le chemin du fichier
    d'origine, first_character_index et last_character_index
    delimitent le morceau dans ce fichier. text contient les
    caracteres compris entre ces deux positions."""

    text: str
