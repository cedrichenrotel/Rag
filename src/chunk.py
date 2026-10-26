from src.model import Chunk


class CutChunk:
    def _chunk_fixed(
        self,
        file_path: str,
        text: str,
        max_chunk_size: int,
        start: int,
        end: int,
        overlap: int = 0,
    ) -> list[Chunk]:
        """Retourne le texte en une liste de x chunk dont la taille est de
        max_chunk_size. utilisation de la fonction min() pour recuperer le bon
        index du dernier caractere.

        overlap: pourcentage (0-99) de max_chunk_size repris au debut du
        chunk suivant (ex: 20 avec 2000 -> 400 caracteres en commun)"""

        chunks: list[Chunk] = []
        step: int = max(1, max_chunk_size - max_chunk_size * overlap // 100)

        for sub_start in range(start, end, step):
            sub_end = min((sub_start + max_chunk_size), end)
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=sub_start,
                    last_character_index=sub_end,
                    text=text[sub_start:sub_end],
                )
            )
            if sub_end == end:
                break
        return chunks

    def _merge_cuts(self, cuts: list[int], max_chunk_size: int) -> list[int]:
        """permet de fusionner des chunks trop petit sans depasser
        max_chunk_size

        Args:
            cuts (list[int]): liste de position d'ou commence chaque chunk
            max_chunk_size (int): nombre d element max que peux contenir un
            chunk

        Returns:
            list[int]: retourne une nouvelle liste cuts modifier
        """
        merged: list[int] = [cuts[0]]
        for i in range(1, len(cuts) - 1):
            if cuts[i + 1] - merged[-1] > max_chunk_size:
                merged.append(cuts[i])
        merged.append(cuts[-1])
        return merged

    def _find_line_positions(self, text: str, suffix: str) -> list[int]:
        """Retourne la position du premier caractere de chaque titre(#)
        fichier.md ou le prefix(class ou def) d'un fichier.py dans le texte."""

        if suffix == ".py":
            prefixes: tuple[str, ...] = ("def ", "class ")
        else:
            prefixes = ("#",)

        positions: list[int] = []
        position: int = 0
        code_bloc: bool = False
        for line in text.splitlines(keepends=True):
            if suffix == ".py":
                to_check = line.lstrip()
            else:
                to_check = line
                if to_check.startswith("```"):
                    code_bloc = not code_bloc

            if to_check.startswith(prefixes) and code_bloc is False:
                positions.append(position)
            position += len(line)

        return positions

    def chunk_file(
        self,
        file_path: str,
        text: str,
        max_chunk_size: int,
        suffix: str,
        overlap: int = 0,
    ) -> list[Chunk]:
        """Decoupe un texte en chunks, un par section. Une section
        commence a un titre '#' (Markdown/texte) ou a un 'def'/'class'(Python),
        selon le suffixe du fichier."""

        cuts: list[int] = self._find_line_positions(text, suffix)
        if not cuts or cuts[0] != 0:
            cuts.insert(0, 0)
        cuts.append(len(text))

        chunks: list[Chunk] = []
        if suffix != ".py":
            cuts = self._merge_cuts(cuts, max_chunk_size)
        for i in range(len(cuts) - 1):
            start = cuts[i]
            end = cuts[i + 1]
            chunks.extend(
                self._chunk_fixed(
                    file_path, text, max_chunk_size, start, end, overlap
                )
            )
        return chunks
