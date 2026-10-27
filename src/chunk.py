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
        """Split the text into a list of chunks whose size is at most
        max_chunk_size. min() is used to get the right index of the
        last character.

        overlap: percentage (0-99) of max_chunk_size repeated at the start
        of the next chunk (e.g. 20 with 2000 -> 400 characters in common)"""

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
        """Merge chunks that are too small without exceeding
        max_chunk_size.

        Args:
            cuts (list[int]): list of positions where each chunk starts
            max_chunk_size (int): maximum number of characters a chunk
            can contain

        Returns:
            list[int]: a new, modified list of cuts
        """
        merged: list[int] = [cuts[0]]
        for i in range(1, len(cuts) - 1):
            if cuts[i + 1] - merged[-1] > max_chunk_size:
                merged.append(cuts[i])
        merged.append(cuts[-1])
        return merged

    def _find_line_positions(self, text: str, suffix: str) -> list[int]:
        """Return the position of the first character of each title (#)
        in a .md file, or of each prefix (class or def) in a .py file."""

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
        """Split a text into chunks, one per section. A section
        starts at a '#' title (Markdown/text) or at a 'def'/'class' (Python),
        depending on the file suffix."""

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
