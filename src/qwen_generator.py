from transformers import AutoTokenizer, AutoModelForCausalLM


class QwenGenerator:
    """permet de lire les chunk de bm25 a Qwen et de renvoyer sa reponse
    dans le fichier answer"""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """charge le model Qwen et son tokenizer pour la generation de texte
        en une seul fois

        Args:
            model_name (str, optional): model de langage. Defaults to
            "Qwen/Qwen3-0.6B".
        """

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_text(self, question: str, max_new_tokens: int = 100) -> str:
        """
            genere le resultat de answer(reponse generer pqr un model de
            langage)

        Args:
            question (str): question soumis a Qwen
            max_new_tokens (int, optional): nombre de token max pour
            la reponse. Defaults to 100.

        Returns:
            str: retourne la reponse de Qwen sans la question
        """

        prompt = self.tokenizer.a

        return


if __name__ == "__main__":
    print()
