from typing import Any

from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding


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

    def _build_context(
        self, list_text: list[str], max_context_tokens: int = 1000
    ) -> str:
        """Assemble les textes des chunks en un seul contexte pour Qwen,
        sans depasser un budget de tokens.

        Les textes sont ajoutes dans l'ordre (le meilleur chunk BM25 en
        premier). Des qu'un texte ferait depasser le budget, on s'arrete :
        on garde ainsi les chunks les plus pertinents sans les couper.
        Exception: si le 1er chunk depasse seul le budget, on garde ses
        max_context_tokens premiers tokens, pour que Qwen ait toujours
        un contexte.

        Args:
            list_text (list[str]): textes des chunks trouves par BM25
            (chunk.text), du plus pertinent au moins pertinent.
            max_context_tokens (int): nombre maximum de tokens du contexte.
            Defaults to 1000.

        Returns:
            str: les textes gardes, separes par une ligne vide.
        """

        conserve_text: list[str] = []
        nb_token_total: int = 0
        for text in list_text:
            nb_token_total += len(self.tokenizer(text)["input_ids"])
            if nb_token_total > max_context_tokens:
                break
            conserve_text.append(text)
        if not conserve_text and list_text:
            input_ids: list[int] = self.tokenizer(list_text[0])["input_ids"]
            cut: list[int] = input_ids[:max_context_tokens]
            decode: str = self.tokenizer.decode(cut, skip_special_tokens=True)
            conserve_text.append(decode)
        context: str = "\n\n".join(conserve_text)
        return context

    def generate_answer(
        self, question: str, list_texts: list[str], max_new_tokens: int = 100,
        max_context_tokens: int = 1000
    ) -> str:
        """
            genere le resultat de answer(reponse generer pqr un model de
            langage)

        Args:
            question (str): question soumis a Qwen
            max_new_tokens (int, optional): nombre de token max pour
            la reponse. Defaults to 100.
            list_texts (list[str]): textes des chunks trouves par BM25,
            du plus pertinent au moins pertinent.
            max_context_tokens (int, optional): nombre maximum de tokens
            du contexte donne a Qwen. Defaults to 1000.

        Returns:
            str: retourne la reponse de Qwen sans la question
        """

        context: str = self._build_context(list_texts, max_context_tokens)
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
        prompt_text: str = self.tokenizer.apply_chat_template(
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
