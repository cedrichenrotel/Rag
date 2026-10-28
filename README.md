*This project has been created as part of the 42 curriculum by cehenrot.*

# Description

## Why a RAG?

Some models, like Qwen3-0.6B, are small and have limited knowledge:
they do not know anything released after their training. If you ask them a
precise question whose answer is not in their data, they make one up:
this is a **hallucination**.

RAG (Retrieval-Augmented Generation) reduces this problem: it lets the model
answer from local data, by giving it the useful passages at the moment
it answers. This is done in three steps.

## Step 1 — Sorting and storing the data

- **Pre-filtering**: we keep all `.py`, `.md` and `.txt` files, which may contain
  useful information to answer the questions.
- **Splitting (chunks)**: the content of each file is split into pieces of
  at most 2000 characters (*chunks*).
  - `.py` files: split at each `def ` and `class `;
  - `.md` / `.txt` files: split at each `#` heading.

  If a file has none of these prefixes, it forms a single section, which is
  only split again if it exceeds 2000 characters.
- **Saving**: the chunks are saved in `data/processed/index.json`, each with:
  - the file path;
  - the position of the first and last character;
  - the text of the passage.

## Step 2 — Search

- **Tokens**: the text of each chunk and the question are lowercased and split
  into a list of words (the *tokens*). This makes it possible to compare the words of the
  question with those of each chunk.
- **Index**: we use the `rank_bm25` library (`BM25Okapi`). It receives the list
  of tokens of each chunk, then, for a question, gives a score to each chunk
  (`get_scores`). The more words of the question a chunk contains, the higher its score.
  But not all words are worth the same: a word that is rare in the corpus (e.g. `lora`)
  earns a lot of points, a word found everywhere (e.g. `the`, `self`) almost nothing.
- **Sorting**: `list_scores` pairs each score with its chunk and sorts them from highest to
  lowest, then `search` keeps the first k.

## Step 3 — Generation

To give the best answer, the language model needs an instruction, a context
and the question.

- **Prompt**: a short and precise instruction so that the model understands what is expected:
  - answer the question concisely and precisely;
  - do not make up the answer;
  - answer "There is no information in the context to answer the question" if the
    context does not contain the necessary information.
- **Context**: the chunks sorted by BM25 are sent to the model as a bundle of
  at most 1000 tokens (to limit the response time). If the first chunk alone exceeds
  this limit, it is truncated.
- **Question**: the question is sent to Qwen together with the context.
- **Ollama**: `OllamaGenerator` uses the same model (`qwen3:0.6b`) as `QwenGenerator`,
  but through the Ollama software, which is much faster (~3 s per question versus ~10 s with
  transformers). Ollama uses a compressed version of the model (*quantization*: each
  weight is stored on fewer bits), which takes less memory and computes faster.
  It is used when the model is asked to answer a large number of questions
  (`answer_dataset`).

# Instructions

Requirements: Python ≥ 3.10, uv, Linux.

## Installation

```bash
make install
```

Installs the dependencies (`uv sync`), downloads Ollama into `~/.local/bin`, starts the
server and downloads the `qwen3:0.6b` model.

## Ollama

```bash
ollama serve   # starts the server (already done by make install)
pkill ollama   # stops the server
```

If the server is stopped, `answer_dataset` uses `QwenGenerator` (transformers), which is slower.

## Pipeline (in order)

Splits `data/raw` into chunks and saves the index in `data/processed/index.json`:

```bash
uv run python -m src index --max_chunk_size 2000
```

Finds the k best chunks for each question and saves the result as JSON:

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions
```

Generates an answer for each question from the retrieved sources:

```bash
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Computes the recall@k (1, 3, 5, 10):

```bash
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10
```

## Makefile shortcuts

- `make index`, `make search_dataset`, `make answer_dataset`, `make evaluate`
  (`DATASET=code` for the code dataset, e.g. `make evaluate DATASET=code`);
- quality: `make lint`, `make lint-strict`, `make clean`.

# Resources

## References

- Official project subject (`en.subject_RAG.pdf`, v2.0).
- BM25: [Okapi BM25 (Wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25) and the
  [rank_bm25](https://github.com/dorianbrown/rank_bm25) library.
- [Pydantic](https://docs.pydantic.dev/), [Python Fire](https://github.com/google/python-fire),
  [tqdm](https://tqdm.github.io/).
- Generation: [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B),
  [transformers](https://huggingface.co/docs/transformers), [Ollama](https://ollama.com/).

## AI usage

I used Claude (Anthropic) as a **mentor**, not as a code generator:

- **Structuring the project**: splitting the work into steps (chunking, index, search,
  generation, robustness), with one concept and one test at each step.
- **Understanding when I was stuck**: explanations of the concepts (RAG, BM25, tokens, recall,
  IoU), and line-by-line walkthroughs of the values to find my mistakes.
- **Reviewing my code**: spotting the cases that caused a traceback or flake8/mypy
  errors, which I then fixed myself.
- **README formatting**: spelling correction, Markdown layout and
  translation into English.

The code was written by me, and I am able to explain and modify every part of it.

# System architecture

## Data flow

```
data/raw/vllm-0.10.1
      │  ChunkCorpus.chunk_corpus (corpus.py) + CutChunk.chunk_file (chunk.py)
      ▼
list[Chunk] ──► PathFile.save ──► data/processed/index.json        (index)
      │
      │  PathFile.load + Bm25Index (bm25_index.py, Tokenizer)
      ▼
top-k Chunk ──► MinimalSource ──► StudentSearchResults (JSON)       (search / search_dataset)
      │
      │  _read_source (re-reads data/raw) + QwenGenerator / OllamaGenerator
      ▼
MinimalAnswer ──► StudentSearchResultsAndAnswer (JSON)              (answer / answer_dataset)
```

## Files in `src/`

- `__main__.py`: entry point, starts the CLI with `Fire(RagCli)`.
- `rag_cli.py`: `RagCli`, the 6 commands (`index`, `search`, `search_dataset`, `answer`,
  `answer_dataset`, `evaluate`).
- `rag_cli_utils.py`: `RagCliUtils`, parent class of `RagCli` with the functions shared
  by the commands (checking `query` and `k`, loading the BM25 index, choosing the
  generator, re-reading a source, display).
- `model.py`: pydantic models exchanged between the steps (`MinimalSource`, `Chunk`,
  `RagDataset`, `StudentSearchResults`…).
- `corpus.py`: `ChunkCorpus`, lists the useful files of `data/raw`, reads them without
  modifying the text and sends them to the splitting step.
- `chunk.py`: `CutChunk`, splits a file into chunks (at `def `/`class ` for `.py`,
  at `#` for `.md`/`.txt`, re-split at `max_chunk_size`).
- `path.py`: `PathFile`, reading and writing JSON files (index, datasets, results).
- `tokenizer.py`: `Tokenizer`, turns a text into a list of words (lowercase,
  snake_case, camelCase).
- `bm25_index.py`: `Bm25Index`, builds BM25 on the chunk tokens (+ words of the
  `file_path`), scores and sorts the chunks.
- `qwen_generator.py`: `QwenGenerator`, builds the context (1000 tokens max) and generates
  the answer with transformers.
- `ollama_generator.py`: `OllamaGenerator`, inherits from `QwenGenerator`, generates the answer
  through the Ollama server (faster).
- `evaluate.py`: `Evaluator`, computes the recall@k (IoU between retrieved sources and expected sources).

# Chunking strategy

## Why split?

The goal is to return the passage that answers the question, not the whole source:

- **Subject limit**: a source longer than 2000 characters makes the whole results
  file invalid for the moulinette.
- **Score (recall)**: a result counts if its character range overlaps the expected
  passage (IoU ≥ 0.05). With a whole file, the overlap is tiny compared to
  the total size: the IoU drops below 0.05, even if it is the right file.
- **Generation**: sending the whole source to Qwen would make the answer too long to generate
  and would exceed the context limit.

## Python files (`.py`): split at `def` / `class`

`class` marks the start of a class and `def` the start of a function or method.
So a chunk usually contains a whole function: a single idea, which keeps
the words of each chunk consistent. The function name is in the chunk, which
helps BM25 when the question is about a specific function.

## Markdown / text files (`.md`, `.txt`): split at `#` headings

Documentation is structured by headings: each `#` opens a section that deals
with a single topic, so one section = one consistent chunk.
The `#` inside a ```` ``` ```` code block are ignored: they are comments
(e.g. `# install vllm` in a bash block), not headings. Splitting there would separate a command
from its heading.

## Sections that are too long

If a section exceeds `max_chunk_size` (2000 by default), it is re-split into
fixed-size pieces (`_chunk_fixed`), with an overlap between them.

## Overlap

The overlap (`index --overlap`, 20 by default) is a percentage of `max_chunk_size` that is
repeated from the previous chunk at the start of the next one. It keeps some context
between two pieces of the same section, which improves the BM25 scoring.

It is computed as follows:

- `max_chunk_size * overlap = x`
- `x // 100 = overlap size`
- `max_chunk_size - overlap size = step`

E.g. with `max_chunk_size = 2000` and `overlap = 20`: overlap size = 400, step = 1600.
A 4500-character section gives 3 chunks: `0-2000`, `1600-3600` and `3200-4500`
(`min()` prevents the last one from going past the end of the section).

Measured recall@5 (overlap 0 → overlap 20):

- public datasets: docs 81.0 % → 82.0 %, code 71.7 % → 72.7 %;
- private datasets: docs 79.0 % → 81.0 %, code 63.0 % → 61.0 %.

The overlap brings docs above the 80 % threshold on the private dataset, so it is kept
(20 by default).

## Text never modified

The text is read and split without any modification (no `strip()`, no line
removal): this keeps the correct position of each character (`first_character_index`,
`last_character_index`) in the source file, so that the moulinette finds the right passage.

# Retrieval method

The search is lexical: it compares the words of the question with the words of each chunk,
using BM25 (`BM25Okapi` from `rank_bm25`), described in the Description.

## Tokenization adapted to code

- **snake_case**: `load_lora_adapter` becomes `load`, `lora`, `adapter`. A question is
  written in separate words ("How to load a LoRA adapter?"): without this split, the chunk
  would have no word in common with it and would get 0 points.
- **camelCase**: `loadLoraAdapter` is split the same way. The rule
  `([a-z]{2})([A-Z])` only splits if there are at least 2 lowercase letters before the uppercase one:
  `LoRA` stays `lora` instead of being split into `lo` and `ra`.
  Measured gain: code 60.6 % → 61.6 %, docs 80.0 % → 82.0 %.
- **Lowercase**: `LoRA`, `Lora` and `lora` become the same token.

## Words from the file path

The tokens of the `file_path` are added to the tokens of each chunk (`tokenize_text`), because
the path often contains keywords: a function in `vllm/lora/models.py` does not necessarily
contain the word "lora", but its path does. The chunk text itself is not modified.
This is the biggest measured gain: code 61.6 % → 68.7 %, docs 82.0 % → 83.0 %.

# Performance analysis

## Recall@5

- **Docs** (100 questions): **82.0 %** public, **81.0 %** private (subject threshold: 80 %).
- **Code** (99 / 100 questions): **72.7 %** public, **61.0 %** private (subject threshold: 50 %).

Progress measured after each improvement (one at a time):

- first version (BM25 + snake_case + camelCase): code 60.6 %, docs 80.0 %;
- camelCase rule `[a-z]{2}` (`LoRA` stays whole): code 61.6 %, docs 82.0 %;
- `file_path` words added to the tokens: code 68.7 %, docs 83.0 %;
- BM25 `k1=1` (instead of 1.5): code 71.7 %, docs 83.0 %;
- overlap 20 (see *Overlap*): code 72.7 %, docs 82.0 %.

Rejected attempts:

- `b=0.5`: code 68.7 %; `b=0.9`: code 70.7 % (docs unchanged, 83.0 %);
- section heading added to the tokens of each chunk: docs 84.0 % but code 69.7 %
  (only for `.md`/`.txt`: docs 83.0 %, code 70.7 %).

## Time

- **Indexing** of the whole corpus (1,969 files, 26,790 chunks): ~3 s (limit: 5 min).
- **Search**: 100 docs questions in ~11 s, 99 code questions in ~13 s
  (limit: 90 s for 200 questions).
- **Generation**: ~3 s per question with Ollama, ~10 s with transformers (CPU).

## Reproducing the measurements

```bash
time uv run python -m src index                      # indexing time
make search_dataset && make evaluate                 # search + docs recall@5
make search_dataset DATASET=code && make evaluate DATASET=code   # search + code recall@5
```

# Design decisions

## BM25 rather than TF-IDF

BM25 limits the effect of repeated words and long chunks:

- **Saturation** (parameter `k1`): a word repeated 50 times does not earn 50 times more
  points; after a few occurrences, the score barely increases.
- **Length** (parameter `b`): a long chunk contains more words by chance, so its
  score is reduced compared to a short chunk.

## Index saved as JSON (not pickle)

- **Readable**: you can open `index.json` and check a chunk by hand.
- **Safe**: `pickle.load` can execute code hidden in a file, `json.load` only
  reads data.
- **Simple with pydantic**: `model_dump` to save, `model_validate` to reload
  and validate the data.

## Ollama with a fallback to transformers

If the Ollama server is not installed or not running, `_choose_generator` switches to
`QwenGenerator` (transformers): no crash, and an answer is still generated with
the same Qwen3-0.6B model, just more slowly.

## Ollama only in `answer_dataset`

Ollama is mostly useful when there are many questions: for 100 questions, ~5 min
with Ollama versus ~17 min with transformers. For a single question (`answer`), the difference
is small.

## Errors raised in the modules, caught in the CLI

`path.py` and `corpus.py` raise errors (`raise`) without catching them, because they do not
know the context. It is `rag_cli.py` that decides what to do depending on the command:
stop (`return`) if the dataset of `search_dataset` cannot be found, or continue
(`continue`) if a source of `answer_dataset` cannot be found.

## `return` rather than `sys.exit(1)`

On error, the command prints a clear message then does `return`. The subject only
requires that no invalid input causes a traceback: `return` is enough and keeps the
code simple.

# Challenges faced

## Building the Qwen prompt (`apply_chat_template`)

The hardest part to understand. Qwen was trained on conversations in a
precise format: `apply_chat_template` turns the `messages` list (`system` and
`user` roles) into a text surrounded by special tags:

```
<|im_start|>system
Answer...<|im_end|>
<|im_start|>user
Context: ... Question: ...<|im_end|>
<|im_start|>assistant
<think>

</think>
```

- `add_generation_prompt=True` adds `<|im_start|>assistant`: it is the model's turn to answer.
- `enable_thinking=False` adds an empty `<think>` block: Qwen3 answers directly, without
  long reasoning.
- `generate` returns the prompt followed by the answer: we cut the first `len_prompt`
  tokens to keep only the answer.

## Generation too slow

With transformers on CPU, an answer takes ~10 s. Solution: Ollama (compressed version
of the same model), ~3 s per question, with a fallback to transformers if Ollama is not running.

## `LoRA` split in two by the camelCase rule

A rule that split before every uppercase letter turned `LoRA` into `lo` + `ra`, which no
longer matched the word `lora` in the question. Solution: split only after at
least 2 lowercase letters (`[a-z]{2}`).

## `#` in Markdown code blocks

Bash comments (`# install vllm`) in ```` ``` ```` blocks were taken for
headings and split the sections. Solution: a boolean `code_bloc`, toggled at each
```` ``` ````, which ignores the `#` inside.

## Progress bar that multiplied the searches

First attempt at tqdm in `search_dataset`: the tqdm loop was added above the
existing loop instead of replacing it, which made 100 × 100 searches. Solution:
wrap the existing questions loop with `tqdm(...)`.

# Example usage

## Search (`search`)

```bash
uv run python -m src search "How to load a LoRA adapter?" --k 3
```

```
Query: How to load a LoRA adapter?

        __ RANKINGS __

Top 1: data/raw/vllm-0.10.1/docs/features/lora.md
    - First_character_index: 4695
    - Last_character_index: 6100
Top 2: data/raw/vllm-0.10.1/tests/entrypoints/openai/test_serving_models.py
    - First_character_index: 2000
    - Last_character_index: 4000
Top 3: data/raw/vllm-0.10.1/vllm/model_executor/model_loader/tensorizer.py
    - First_character_index: 28653
    - Last_character_index: 30571
```

## Answer (`answer`)

```bash
uv run python -m src answer "How to load a LoRA adapter?" --k 3
```

Prints the same ranking, then Qwen's answer:

```
     __ GENERATE ANSWER __

    - Answer: To load a LoRA adapter, send a POST request to the `/v1/load_lora_adapter`
      endpoint with the name and path to the LoRA adapter.
```

## Invalid inputs (no traceback)

```bash
uv run python -m src search "" --k 3
uv run python -m src search "LoRA" --k 0
uv run python -m src index --raw_path nope
```

```
[ERROR] query must be a non-empty string.
[ERROR] k must be an integer >= 1 (got 0)
Index: [WARNING] list_files: Not a directory: nope
```
