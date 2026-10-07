from src.model import Chunk


class CutChunk:
    def chunk_fixed(
        self,
        file_path: str,
        text: str,
        max_chunk_size: int,
        start: int,
        end: int,
    ) -> list[Chunk]:
        """Retourne le texte en une liste de x chunk dont la taille est de
        max_chunk_size. utilisation de la fonction min() pour recuperer le bon
        index du dernier caractere"""

        chunks: list[Chunk] = []

        for sub_start in range(start, end, max_chunk_size):
            sub_end = min((sub_start + max_chunk_size), end)
            chunks.append(
                Chunk(
                    file_path=file_path,
                    first_character_index=sub_start,
                    last_character_index=sub_end,
                    text=text[sub_start:sub_end],
                )
            )
        return chunks

    def find_line_positions(self, text: str, suffix: str) -> list[int]:
        """Retourne la position du premier caractere de chaque titre(#)
        fichier.md ou le prefix(class ou def) d'un fichier.py dans le texte."""

        if suffix == ".py":
            prefixes: tuple[str, ...] = ("def ", "class ")
        else:
            prefixes = ("#",)

        positions: list[int] = []
        position: int = 0

        for line in text.splitlines(keepends=True):
            if suffix == ".py":
                to_check = line.lstrip()
            else:
                to_check = line

            if to_check.startswith(prefixes):
                positions.append(position)
            position += len(line)

        return positions

    def chunk_file(
        self, file_path: str, text: str, max_chunk_size: int, suffix: str
    ) -> list[Chunk]:
        """Decoupe un texte en chunks, un par section. Une section
        commence a un titre '#' (Markdown/texte) ou a un 'def'/'class'(Python),
        selon le suffixe du fichier."""

        cuts: list[int] = self.find_line_positions(text, suffix)
        if not cuts or cuts[0] != 0:
            cuts.insert(0, 0)
        cuts.append(len(text))

        chunks: list[Chunk] = []
        for i in range(len(cuts) - 1):
            start = cuts[i]
            end = cuts[i + 1]
            chunks.extend(
                self.chunk_fixed(file_path, text, max_chunk_size, start, end)
            )

        return chunks


if __name__ == "__main__":
    cutchunk = CutChunk()

    text = "import os\n\nclass Engine:\n    def start(self):\n        pass\n"
    for c in cutchunk.chunk_file("test.py", text, 2000, ".py"):
        print(c.first_character_index, c.last_character_index, repr(c.text))

    text = "# A\n" + "a" * 100 + "\n# B\n" + "b" * 50
    for c in cutchunk.chunk_file("test.md", text, 2000, ".md"):
        print(
            c.first_character_index, c.last_character_index, repr(c.text[:8])
        )
    # print(line)
    # print(len(line))
