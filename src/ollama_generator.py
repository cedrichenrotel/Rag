from typing import Any

import ollama
from transformers import AutoTokenizer

from src.qwen_generator import QwenGenerator


class OllamaGenerator(QwenGenerator):
    """permet de generer les reponses comme QwenGenerator via Ollama.
    On envois le prompt au serveur Ollama au lieu d'utiliser transformers.
    Ollama est une version compresser de Qwen qui reduit le temp de reponse"""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        model_ollama: str = "qwen3:0.6b",
    ) -> None:
        """Charge les tokanizer pour la preparation d'Ollama

        Args:
            model_name (str, optional): nom du model utiliser pour le
            tokenizer. Defaults to "Qwen/Qwen3-0.6B".
            model_ollama (_type_, optional): nom du model cote server Ollama.
            Defaults to "qwen3:0.6b".
        """
        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model_ollama: str = model_ollama

    def generate_answer(
        self,
        question: str,
        list_texts: list[str],
        max_new_tokens: int = 100,
        max_context_tokens: int = 1000,
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
        reponse: ollama.ChatResponse = ollama.chat(
            model=self.model_ollama,
            messages=messages,
            think=False,
            options={"num_predict": max_new_tokens},
        )
        if not reponse.message.content:
            generate_answer: str = ""
        else:
            generate_answer = reponse.message.content

        return generate_answer
