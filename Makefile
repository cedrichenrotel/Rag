UV := uv
OLLAMA := $(HOME)/.local/bin/ollama
OLLAMA_DIR := $(HOME)/.ollama/models
MAX_CHUNK := 2000
RAW_PATH := data/raw
INDEX_PATH := data/processed/index.json
DATASET := docs
UNANSWERED := data/datasets/UnansweredQuestions/dataset_$(DATASET)_public.json
ANSWERED := data/datasets/AnsweredQuestions/dataset_$(DATASET)_public.json
SEARCH_DIR := data/output/search_results/UnansweredQuestions
ANSWER_DIR := data/output/search_results_and_answer/UnansweredQuestions
SEARCH_RESULT := $(SEARCH_DIR)/dataset_$(DATASET)_public.json
K := 10

.PHONY: install run debug clean lint lint-strict index search_dataset evaluate answer_dataset

install:
	$(UV) sync
	@if [ ! -x $(OLLAMA) ]; then \
		mkdir -p $(HOME)/.local && \
		curl -fsSL https://ollama.com/download/ollama-linux-amd64.tar.zst \
		| tar --zstd -x -C $(HOME)/.local; \
	fi
		OLLAMA_MODELS=$(OLLAMA_DIR) $(OLLAMA) serve > /dev/null 2>&1 & sleep 3; $(OLLAMA) pull qwen3:0.6b
run:
	@$(UV) run python -m src index --max_chunk_size $(MAX_CHUNK)

$(INDEX_PATH):
	@$(UV) run python -m src index --max_chunk_size $(MAX_CHUNK) --raw_path $(RAW_PATH) --index_path $(INDEX_PATH)

index:
	@$(UV) run python -m src index --max_chunk_size $(MAX_CHUNK) --raw_path $(RAW_PATH) --index_path $(INDEX_PATH)

search_dataset: $(INDEX_PATH)
	@$(UV) run python -m src search_dataset --dataset_path $(UNANSWERED) --save_directory $(SEARCH_DIR) --k $(K) --index_path $(INDEX_PATH)

answer_dataset:
	@$(UV) run python -m src answer_dataset --student_search_results_path $(SEARCH_RESULT) --save_directory $(ANSWER_DIR)

evaluate:
	@$(UV) run python -m src evaluate --student_search_results_path $(SEARCH_RESULT) --dataset_path $(ANSWERED) --k $(K)

debug:
	$(UV) run python -m pdb -m src index --max_chunk_size $(MAX_CHUNK)

clean:
	rm -rf __pycache__ .mypy_cache
	rm -rf dist build *.egg-info
	find . -name "*.pyc" -delete

lint:
	$(UV) run flake8 .
	$(UV) run mypy . --warn-return-any \
	--warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs \
	--check-untyped-defs

lint-strict:
	$(UV) run flake8 .
	$(UV) run mypy . --strict
