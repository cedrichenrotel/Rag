import json
from pathlib import Path
from typing import Any

from src.model import Chunk


class PathFile:
    def create_json_missing(self, file: Path) -> None:
        """permet de creer le dossier complet data/processed.

        file: fichier cibler"""

        file.parent.mkdir(parents=True, exist_ok=True)

    def save(self, chunks: list[Chunk], path: Path) -> None:
        """Sauvegarde et convertis en dict(json), les donnees des
        resultats.

        path: chemin du fichier a creer (ex: data/processed/index.json).
        S'il existe deja, il est remplace.
        data: stock dans sa liste toutes les info que la class chunk a besoin
        pour le stocker dans le fichier JSON. flag(index)"""

        self.create_json_missing(path)

        with open(path, "w") as f:
            data: list[dict[str, Any]] = [
                chunk.model_dump() for chunk in chunks
            ]
            json.dump(data, f)

    def load(self, path: Path) -> list[Chunk]:
        """Permet de retrouver la liste des chunks au moment du 'def save'
        sans relire ni decouper le corpus (flag -> search).

        path: fichier cibler"""

        with open(path, "r") as f:
            data: list[dict[str, Any]] = json.load(f)
            list_chunks: list[Chunk] = [Chunk.model_validate(d) for d in data]
        return list_chunks
