import re


class Tokenizer:
    """Convert the text into a list of
    lowercase words."""

    def tokenize(self, text: str) -> list[str]:
        """Split the text into a list of words
        and convert the characters to lowercase.

        text: text to split into tokens"""

        re_text: str = re.sub(r"([a-z]{2})([A-Z])", r"\1 \2", text)
        text_lowercase: str = re_text.lower()
        return re.findall(r"[a-z0-9]+", text_lowercase)
