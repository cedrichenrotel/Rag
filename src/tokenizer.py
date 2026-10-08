import re


class Tokenizer:
    """Convertis le texte en une liste de mot
    en minuscule"""

    def tokenize(self, text: str) -> list[str]:
        """coupe le texte en une liste de de mot
        et convertis les caracteres en minuscule

        text: contenu d'un token"""

        re_text: str = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
        text_lowercase: str = re_text.lower()
        return re.findall(r"[a-z0-9]+", text_lowercase)


if __name__ == "__main__":
    tok = Tokenizer()
    rst = tok.tokenize("def load_lora_ada,pter(self, LoraAdapter):")
    print(rst)
    print("ok")
