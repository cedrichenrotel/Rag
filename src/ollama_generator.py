from typing import Any

import ollama
from transformers import AutoTokenizer

from src.qwen_generator import QwenGenerator


class OllamaGenerator(QwenGenerator):
    """Generate the answers like QwenGenerator, but through Ollama.
    The prompt is sent to the Ollama server instead of using transformers.
    Ollama runs a compressed version of Qwen that reduces response time."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen3-0.6B",
        model_ollama: str = "qwen3:0.6b",
    ) -> None:
        """Load the tokenizer used to prepare the prompt for Ollama.

        Args:
            model_name (str, optional): name of the model used for the
            tokenizer. Defaults to "Qwen/Qwen3-0.6B".
            model_ollama (str, optional): name of the model on the Ollama
            server. Defaults to "qwen3:0.6b".
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
        """Generate the answer (text written by a language model).

        Args:
            question (str): question given to Qwen
            max_new_tokens (int, optional): maximum number of tokens for
            the answer. Defaults to 100.
            list_texts (list[str]): texts of the chunks found by BM25,
            from the most relevant to the least relevant.
            max_context_tokens (int, optional): maximum number of tokens
            of the context given to Qwen. Defaults to 1000.

        Returns:
            str: Qwen's answer, without the question
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
