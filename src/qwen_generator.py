from typing import Any

from torch import Tensor
from transformers import AutoModelForCausalLM, AutoTokenizer, BatchEncoding


class QwenGenerator:
    """Give the BM25 chunks to Qwen and return its answer."""

    def __init__(self, model_name: str = "Qwen/Qwen3-0.6B") -> None:
        """Load the Qwen model and its tokenizer once, for text
        generation.

        Args:
            model_name (str, optional): language model. Defaults to
            "Qwen/Qwen3-0.6B".
        """

        self.tokenizer: Any = AutoTokenizer.from_pretrained(model_name)
        self.model: Any = AutoModelForCausalLM.from_pretrained(model_name)

    def _build_context(
        self, list_text: list[str], max_context_tokens: int = 1000
    ) -> str:
        """Join the chunk texts into a single context for Qwen,
        without exceeding a token budget.

        The texts are added in order (the best BM25 chunk first). As soon
        as a text would exceed the budget, we stop: this keeps the most
        relevant chunks without cutting them.
        Exception: if the 1st chunk alone exceeds the budget, we keep its
        first max_context_tokens tokens, so that Qwen always has
        a context.

        Args:
            list_text (list[str]): texts of the chunks found by BM25
            (chunk.text), from the most relevant to the least relevant.
            max_context_tokens (int): maximum number of tokens of the
            context. Defaults to 1000.

        Returns:
            str: the kept texts, separated by an empty line.
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
