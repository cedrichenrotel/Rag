*This project has been created as part of the 42 curriculum by cehenrot.*

# Description

## Pourquoi un RAG ?

Certains modèles, comme Qwen3-0.6B, sont petits et leurs connaissances sont limitées :
ils ne connaissent pas ce qui est sorti après leur entraînement. Si on leur pose une
question précise dont la réponse n'est pas dans leurs données, ils en inventent une :
c'est une **hallucination**.

Le RAG (Retrieval-Augmented Generation) réduit ce phénomène : il permet au modèle de
répondre à partir de données locales, en lui donnant les passages utiles au moment
où il répond. Cela se fait en trois étapes.

## Étape 1 — Tri et stockage des données

- **Pré-tri** : on garde tous les fichiers `.py`, `.md` et `.txt`, qui peuvent contenir
  des informations utiles pour répondre aux questions.
- **Découpage (chunks)** : le contenu de chaque fichier est découpé en morceaux de
  2000 caractères maximum (des *chunks*).
  - fichiers `.py` : découpe à chaque `def ` et `class ` ;
  - fichiers `.md` / `.txt` : découpe à chaque titre `#`.

  Si un fichier n'a aucun de ces préfixes, il forme une seule section, qui n'est
  recoupée que si elle dépasse 2000 caractères.
- **Sauvegarde** : les chunks sont sauvegardés dans `data/processed/index.json`, avec pour chacun :
  - le chemin du fichier ;
  - la position du premier et du dernier caractère ;
  - le texte du passage.

## Étape 2 — Recherche

- **Tokens** : le texte de chaque chunk et la question sont mis en minuscules et découpés
  en une liste de mots (les *tokens*). Cela permet ensuite de comparer les mots de la
  question à ceux de chaque chunk.
- **Index** : on utilise la bibliothèque `rank_bm25` (`BM25Okapi`). Elle reçoit la liste
  des tokens de chaque chunk, puis, pour une question, donne une note à chaque chunk
  (`get_scores`). Plus un chunk contient de mots de la question, plus sa note est élevée.
  Mais tous les mots ne valent pas pareil : un mot rare dans le corpus (ex. `lora`)
  rapporte beaucoup de points, un mot présent partout (ex. `the`, `self`) presque rien.
- **Tri** : `list_scores` associe chaque note à son chunk et les trie de la plus haute à
  la plus basse, puis `search` garde les k premiers.

## Étape 3 — Génération

Pour répondre au mieux, le modèle de langage a besoin d'une instruction, d'un contexte
et de la question.

- **Prompt** : une instruction courte et précise pour que le modèle comprenne ce qu'on attend :
  - répondre à la question de manière concise et précise ;
  - ne pas inventer la réponse ;
  - répondre « There is no information in the context to answer the question » si le
    contexte ne contient pas les informations nécessaires.
- **Contexte** : les chunks triés par BM25 sont envoyés au modèle en un paquet de
  1000 tokens maximum (pour limiter le temps de réponse). Si le premier chunk dépasse
  à lui seul cette limite, il est coupé.
- **Question** : la question est envoyée à Qwen avec le contexte.
- **Ollama** : `OllamaGenerator` utilise le même modèle (`qwen3:0.6b`) que `QwenGenerator`,
  mais via le logiciel Ollama, beaucoup plus rapide (~3 s par question contre ~10 s avec
  transformers). Ollama utilise une version compressée du modèle (*quantization* : chaque
  poids est stocké sur moins de bits), qui prend moins de mémoire et se calcule plus vite.
  Il est utilisé quand on demande au modèle de répondre à un grand nombre de questions
  (`answer_dataset`).

# Instructions

Prérequis : Python ≥ 3.10, uv, Linux.

## Installation

```bash
make install
```

Installe les dépendances (`uv sync`), télécharge Ollama dans `~/.local/bin`, lance le
serveur et télécharge le modèle `qwen3:0.6b`.

## Ollama

```bash
ollama serve   # lance le serveur (déjà fait par make install)
pkill ollama   # arrête le serveur
```

Si le serveur est arrêté, `answer_dataset` utilise `QwenGenerator` (transformers), plus lent.

## Pipeline (dans l'ordre)

Découpe `data/raw` en chunks et sauvegarde l'index dans `data/processed/index.json` :

```bash
uv run python -m src index --max_chunk_size 2000
```

Cherche les k meilleurs chunks pour chaque question et sauvegarde le résultat en JSON :

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions
```

Génère une réponse pour chaque question à partir des sources trouvées :

```bash
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

Calcule le recall@k (1, 3, 5, 10) :

```bash
uv run python -m src evaluate \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --dataset_path data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10
```

## Raccourcis Makefile

- `make index`, `make search_dataset`, `make answer_dataset`, `make evaluate`
  (`DATASET=code` pour le dataset code, ex. `make evaluate DATASET=code`) ;
- qualité : `make lint`, `make lint-strict`, `make clean`.

# Resources

## Références

- Sujet officiel du projet (`en.subject_RAG.pdf`, v2.0).
- BM25 : [Okapi BM25 (Wikipedia)](https://en.wikipedia.org/wiki/Okapi_BM25) et la
  bibliothèque [rank_bm25](https://github.com/dorianbrown/rank_bm25).
- [Pydantic](https://docs.pydantic.dev/), [Python Fire](https://github.com/google/python-fire),
  [tqdm](https://tqdm.github.io/).
- Génération : [Qwen3-0.6B](https://huggingface.co/Qwen/Qwen3-0.6B),
  [transformers](https://huggingface.co/docs/transformers), [Ollama](https://ollama.com/).

## Usage de l'IA

J'ai utilisé Claude (Anthropic) comme **mentor**, pas comme générateur de code :

- **Structurer le projet** : découper le travail en étapes (chunking, index, recherche,
  génération, robustesse), avec un concept et un test à chaque étape.
- **Comprendre quand je bloquais** : explication des notions (RAG, BM25, tokens, recall,
  IoU), et déroulé des valeurs ligne par ligne pour trouver mes erreurs.
- **Relire mon code** : repérer les cas qui provoquaient un traceback ou des erreurs
  flake8/mypy, que j'ai ensuite corrigés moi-même.
- **Mise en forme du README** : correction de l'orthographe, mise en page Markdown et
  traduction en anglais.

Le code a été écrit par moi, et je suis capable d'expliquer et de modifier chaque partie.

# System architecture

## Flux des données

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
      │  _read_source (relit data/raw) + QwenGenerator / OllamaGenerator
      ▼
MinimalAnswer ──► StudentSearchResultsAndAnswer (JSON)              (answer / answer_dataset)
```

## Fichiers de `src/`

- `__main__.py` : point d'entrée, lance la CLI avec `Fire(RagCli)`.
- `rag_cli.py` : `RagCli`, les 6 commandes (`index`, `search`, `search_dataset`, `answer`,
  `answer_dataset`, `evaluate`).
- `rag_cli_utils.py` : `RagCliUtils`, classe mère de `RagCli` avec les fonctions communes
  aux commandes (vérification de `query` et `k`, chargement de l'index BM25, choix du
  générateur, relecture d'une source, affichage).
- `model.py` : modèles pydantic échangés entre les étapes (`MinimalSource`, `Chunk`,
  `RagDataset`, `StudentSearchResults`…).
- `corpus.py` : `ChunkCorpus`, liste les fichiers utiles de `data/raw`, les lit sans
  modifier le texte et les envoie au découpage.
- `chunk.py` : `CutChunk`, découpe un fichier en chunks (aux `def `/`class ` pour `.py`,
  aux `#` pour `.md`/`.txt`, recoupe à `max_chunk_size`).
- `path.py` : `PathFile`, lecture et écriture des JSON (index, datasets, résultats).
- `tokenizer.py` : `Tokenizer`, transforme un texte en liste de mots (minuscules,
  snake_case, camelCase).
- `bm25_index.py` : `Bm25Index`, construit BM25 sur les tokens des chunks (+ mots du
  `file_path`), note et trie les chunks.
- `qwen_generator.py` : `QwenGenerator`, construit le contexte (1000 tokens max) et génère
  la réponse avec transformers.
- `ollama_generator.py` : `OllamaGenerator`, hérite de `QwenGenerator`, génère la réponse
  via le serveur Ollama (plus rapide).
- `evaluate.py` : `Evaluator`, calcule le recall@k (IoU entre sources trouvées et sources attendues).

# Chunking strategy

## Pourquoi découper ?

Le but est de renvoyer le passage qui répond à la question, pas toute la source :

- **Limite du sujet** : une source de plus de 2000 caractères rend tout le fichier de
  résultats invalide pour la moulinette.
- **Score (recall)** : un résultat compte si sa plage de caractères chevauche le passage
  attendu (IoU ≥ 0.05). Avec un fichier entier, le chevauchement est minuscule par rapport
  à la taille totale : l'IoU tombe sous 0,05, même si c'est le bon fichier.
- **Génération** : envoyer toute la source à Qwen rendrait la réponse trop longue à générer
  et dépasserait la limite de contexte.

## Fichiers Python (`.py`) : découpe aux `def` / `class`

`class` marque le début d'une classe et `def` le début d'une fonction ou méthode.
Un chunk contient donc en général une fonction entière : une seule idée, ce qui garde
une cohérence dans les mots de chaque chunk. Le nom de la fonction est dans le chunk, ce
qui aide BM25 quand la question porte sur une fonction précise.

## Fichiers Markdown / texte (`.md`, `.txt`) : découpe aux titres `#`

Une documentation est structurée par des titres : chaque `#` ouvre une section qui parle
d'un seul sujet, donc une section = un chunk cohérent.
Les `#` situés dans un bloc de code ```` ``` ```` sont ignorés : ce sont des commentaires
(ex. `# install vllm` dans un bloc bash), pas des titres. Couper là séparerait une commande
de son titre.

## Sections trop longues

Si une section dépasse `max_chunk_size` (2000 par défaut), elle est recoupée en morceaux de
taille fixe (`chunk_fixed`). Ex. : une section de 4500 caractères donne 3 chunks de 2000,
2000 et 500 caractères (`min()` empêche le dernier de dépasser la fin de la section).

## Texte jamais modifié

Le texte est lu et découpé sans aucune modification (pas de `strip()`, pas de suppression
de lignes) : cela garde la bonne position de chaque caractère (`first_character_index`,
`last_character_index`) dans le fichier source, pour que la moulinette retrouve le bon passage.

# Retrieval method

La recherche est lexicale : elle compare les mots de la question aux mots de chaque chunk,
avec BM25 (`BM25Okapi` de `rank_bm25`), décrit dans la Description.

## Tokenisation adaptée au code

- **snake_case** : `load_lora_adapter` devient `load`, `lora`, `adapter`. Une question est
  écrite en mots séparés (« How to load a LoRA adapter? ») : sans ce découpage, le chunk
  n'aurait aucun mot en commun avec elle et aurait 0 point.
- **camelCase** : `loadLoraAdapter` est coupé de la même façon. La règle
  `([a-z]{2})([A-Z])` coupe seulement s'il y a au moins 2 minuscules avant la majuscule :
  `LoRA` reste `lora` au lieu d'être coupé en `lo` et `ra`.
  Gain mesuré : code 60,6 % → 61,6 %, docs 80,0 % → 82,0 %.
- **Minuscules** : `LoRA`, `Lora` et `lora` deviennent le même token.

## Mots du chemin du fichier

Les tokens du `file_path` sont ajoutés aux tokens de chaque chunk (`tokenize_text`), car
le chemin contient souvent des mots clés : une fonction de `vllm/lora/models.py` n'écrit
pas forcément « lora », mais son chemin le dit. Le texte du chunk, lui, n'est pas modifié.
C'est le plus gros gain mesuré : code 61,6 % → 68,7 %, docs 82,0 % → 83,0 %.

# Performance analysis

## Recall@5 (datasets publics)

- **Docs** (100 questions) : **83,0 %** (seuil du sujet : 80 %).
- **Code** (99 questions) : **71,7 %** (seuil du sujet : 50 %).

Évolution mesurée à chaque amélioration (une seule à la fois) :

- première version (BM25 + snake_case + camelCase) : code 60,6 %, docs 80,0 % ;
- règle camelCase `[a-z]{2}` (`LoRA` reste entier) : code 61,6 %, docs 82,0 % ;
- mots du `file_path` ajoutés aux tokens : code 68,7 %, docs 83,0 % ;
- BM25 `k1=1` (au lieu de 1,5) : code 71,7 %, docs 83,0 %.

Essais rejetés :

- `b=0.5` : code 68,7 % ; `b=0.9` : code 70,7 % (docs inchangée, 83,0 %) ;
- titre de section ajouté aux tokens de chaque chunk : docs 84,0 % mais code 69,7 %
  (seulement pour `.md`/`.txt` : docs 83,0 %, code 70,7 %).

## Temps

- **Indexation** du corpus entier (1 969 fichiers, 27 103 chunks) : ~2,4 s (limite : 5 min).
- **Recherche** : 100 questions docs en ~6 s, 99 questions code en ~7 s
  (limite : 90 s pour 200 questions).
- **Génération** : ~3 s par question avec Ollama, ~10 s avec transformers (CPU).

## Reproduire les mesures

```bash
time uv run python -m src index                      # temps d'indexation
make search_dataset && make evaluate                 # recherche + recall@5 docs
make search_dataset DATASET=code && make evaluate DATASET=code   # recherche + recall@5 code
```

# Design decisions

## BM25 plutôt que TF-IDF

BM25 limite l'effet des mots répétés et des chunks longs :

- **Saturation** (paramètre `k1`) : un mot répété 50 fois ne rapporte pas 50 fois plus
  de points ; après quelques occurrences, le score n'augmente presque plus.
- **Longueur** (paramètre `b`) : un chunk long contient plus de mots par hasard, donc son
  score est réduit par rapport à un chunk court.

## Index sauvegardé en JSON (pas pickle)

- **Lisible** : on peut ouvrir `index.json` et vérifier un chunk à la main.
- **Sûr** : `pickle.load` peut exécuter du code caché dans un fichier, `json.load` ne fait
  que lire des données.
- **Simple avec pydantic** : `model_dump` pour sauvegarder, `model_validate` pour recharger
  et vérifier les données.

## Ollama avec repli sur transformers

Si le serveur Ollama n'est pas installé ou pas lancé, `_choose_generator` bascule sur
`QwenGenerator` (transformers) : pas de crash, et une réponse est quand même générée avec
le même modèle Qwen3-0.6B, juste plus lentement.

## Ollama seulement dans `answer_dataset`

Ollama est surtout utile quand il y a beaucoup de questions : pour 100 questions, ~5 min
avec Ollama contre ~17 min avec transformers. Pour une seule question (`answer`), l'écart
est faible.

## Erreurs levées dans les modules, attrapées dans la CLI

`path.py` et `corpus.py` lèvent les erreurs (`raise`) sans les attraper, car ils ne
connaissent pas le contexte. C'est `rag_cli.py` qui décide quoi faire selon la commande :
arrêter (`return`) si le dataset de `search_dataset` est introuvable, ou continuer
(`continue`) si une source de `answer_dataset` est introuvable.

## `return` plutôt que `sys.exit(1)`

En cas d'erreur, la commande affiche un message clair puis fait `return`. Le sujet demande
seulement qu'aucune entrée invalide ne provoque de traceback : `return` suffit et garde le
code simple.

# Challenges faced

## Construire le prompt de Qwen (`apply_chat_template`)

La partie la plus difficile à comprendre. Qwen a été entraîné sur des conversations dans un
format précis : `apply_chat_template` transforme la liste `messages` (rôles `system` et
`user`) en un texte entouré de balises spéciales :

```
<|im_start|>system
Answer...<|im_end|>
<|im_start|>user
Context: ... Question: ...<|im_end|>
<|im_start|>assistant
<think>

</think>
```

- `add_generation_prompt=True` ajoute `<|im_start|>assistant` : c'est au modèle de répondre.
- `enable_thinking=False` ajoute un bloc `<think>` vide : Qwen3 répond directement, sans
  long raisonnement.
- `generate` renvoie le prompt suivi de la réponse : on coupe les `len_prompt` premiers
  tokens pour garder seulement la réponse.

## Génération trop lente

Avec transformers sur CPU, une réponse prend ~10 s. Solution : Ollama (version compressée
du même modèle), ~3 s par question, avec repli sur transformers si Ollama n'est pas lancé.

## `LoRA` coupé en deux par la règle camelCase

Une règle qui coupe avant chaque majuscule transformait `LoRA` en `lo` + `ra`, qui ne
correspondaient plus au mot `lora` de la question. Solution : couper seulement après au
moins 2 minuscules (`[a-z]{2}`).

## `#` dans les blocs de code Markdown

Les commentaires bash (`# install vllm`) dans les blocs ```` ``` ```` étaient pris pour des
titres et coupaient les sections. Solution : un booléen `code_bloc`, basculé à chaque
```` ``` ````, qui ignore les `#` à l'intérieur.

## Barre de progression qui multipliait les recherches

Premier essai de tqdm dans `search_dataset` : la boucle tqdm était ajoutée au-dessus de la
boucle existante au lieu de la remplacer, ce qui faisait 100 × 100 recherches. Solution :
envelopper la boucle des questions existante avec `tqdm(...)`.

# Example usage

## Recherche (`search`)

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

## Réponse (`answer`)

```bash
uv run python -m src answer "How to load a LoRA adapter?" --k 3
```

Affiche le même classement, puis la réponse de Qwen :

```
     __ GENERATE ANSWER __

    - Answer: To load a LoRA adapter, send a POST request to the `/v1/load_lora_adapter`
      endpoint with the name and path to the LoRA adapter.
```

## Entrées invalides (pas de traceback)

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
