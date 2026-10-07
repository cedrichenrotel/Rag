from src.model import Chunk


def chunk_fixed(file_path: str, text: str, max_chunk_size: int) -> list[Chunk]:
    """Retourne le texte en une liste de x chunk dont la taille est de
    max_chunk_size. utilisation de la fonction min() pour recuperer le bon
    index du dernier caractere"""

    chunks: list[Chunk] = []

    for start in range(0, len(text), max_chunk_size):
        end = start + max_chunk_size
        end = min(end, len(text))
        chunks.append(
            Chunk(
                file_path=file_path,
                first_character_index=start,
                last_character_index=end,
                text=text[start:end],
            )
        )
    return chunks


def find_heading_positions(text: str) -> list[int]:
    """Retourne la position du premier caractere de chaque titre
    Markdown (ligne qui commence par '#') dans le texte."""

    positions: list[int] = []
    position: int = 0

    for line in text.splitlines(keepends=True):
        if line.startswith("#"):
            positions.append(position)
        position += len(line)

    return positions


def chunk_markdown(
    file_path: str, text: str, max_chunk_size: int
) -> list[Chunk]:
    """Decoupe un texte Markdown en chunks, un par section
    (un titre et le texte qui le suit)."""

    cuts: list[int] = find_heading_positions(text)
    if cuts[0] != 0:
        cuts.insert(0, 0)
    cuts.append(len(text))

    chunks: list[Chunk] = []
    for i in range(len(cuts) - 1):
        start = cuts[i]
        end = cuts[i + 1]
        chunks.append(
                    Chunk(
                        file_path=file_path,
                        first_character_index=start,
                        last_character_index=end,
                        text=text[start:end],
                    )
                )
    return chunks
