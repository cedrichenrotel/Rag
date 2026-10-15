from typing import Any
from torch import Tensor
from transformers import AutoTokenizer, AutoModelForCausalLM, BatchEncoding


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

        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(model_name)

    def generate_answer(
        self, question: str, context: str, max_new_tokens: int = 100
    ) -> str:
        """
            genere le resultat de answer(reponse generer pqr un model de
            langage)

        Args:
            question (str): question soumis a Qwen
            max_new_tokens (int, optional): nombre de token max pour
            la reponse. Defaults to 100.
            context: texte brut trouver par BM25

        Returns:
            str: retourne la reponse de Qwen sans la question
        """

        prompt: str = (
            "Answer the question concisely and as accurately "
            "as possible, using only the context. "
            "If the context does not contain the answer, "
            "reply exactly: 'There is no information in the "
            "context to answer the question'."
        )
        messages: list[dict[str, str]] = [
            {"role": "system", "content": prompt},
            {
                "role": "user",
                "content": f"Context:\n{context}\n\nQuestion:\n{question}",
            },
        ]
        prompt_text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        if not isinstance(prompt_text, str):
            raise TypeError(
                f"[ERROR] generate_answer(): Typing error: {type(prompt_text)}"
            )
        input_ids: BatchEncoding = self.tokenizer(
            prompt_text, return_tensors="pt"
        )
        result_generate: Tensor = self.model.generate(
            **input_ids,
            max_new_tokens=max_new_tokens,
        )
        len_prompt: int = len(input_ids["input_ids"][0])
        answer_generate: Tensor = result_generate[0][len_prompt:]
        decode_answer: str = self.tokenizer.decode(
            answer_generate, skip_special_tokens=True
        )
        return decode_answer
