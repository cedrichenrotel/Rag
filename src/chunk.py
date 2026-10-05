from src.model import Chunk


def chunk_fixed(file_path: str, text: str, max_chunk_size: int) -> list[Chunk]:
    """Retourne le text en une liste de x chunk dont la taille est de
    max_chunk_size"""

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
