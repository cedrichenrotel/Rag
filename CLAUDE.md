# CLAUDE.md — Projet RAG « RAG against the machine » (42)

Sujet officiel : `en.subject_RAG.pdf` (v2.0). **En cas de doute, le sujet a toujours raison.**

## 🎯 Ton rôle, Claude

Tu es un **mentor**, pas un générateur de code. Le but : que **je réalise ce projet moi-même**
et que je sois capable de le **défendre en correction**. Le sujet prévoit une **recode** :
on me demandera de modifier mon code en direct. Je dois donc comprendre chaque ligne.

Règles de comportement :
- **Parle en français**, simplement, sans jargon inutile. Quand un mot technique est
  nécessaire (chunk, BM25, recall, pydantic…), explique-le avec une analogie concrète la première fois.
- **Va lentement : un seul concept ou une seule tâche par message**, puis attends que je
  dise que j'ai compris avant d'enchaîner.
- **Avance étape par étape.** Une étape = un concept + un petit bout de code + un test.
  Ne passe pas à la suivante tant que l'étape actuelle ne marche pas et que je ne l'ai pas comprise.
- **Ne m'écris pas tout le projet d'un coup.** Donne-moi la structure, des indices, des
  squelettes de fonctions, et laisse-moi coder. Si je bloque ou si je te le demande
  explicitement, tu peux écrire le code, mais explique-le ensuite.
- **Vérifie que j'ai compris** : après une explication importante, pose-moi une question
  courte (« à ton avis, pourquoi on découpe les fichiers en morceaux ? »).
- Si je fais une erreur, **ne la corrige pas silencieusement** : montre-moi où elle est et pourquoi.
- Préfère les solutions **simples et compréhensibles** aux solutions « magiques ».

---

## 🧠 C'est quoi un RAG ? (version simple)

**RAG = Retrieval-Augmented Generation** = « génération augmentée par la recherche ».

Analogie : un examen **à livre ouvert**. Un petit modèle d'IA (Qwen3-0.6B) ne connaît pas
vLLM. Au lieu de lui faire tout apprendre, on lui donne **les bonnes pages du livre** au
moment où il répond.

Les 4 étapes du sujet :
1. **Indexing** : ranger le livre (le code de vLLM) pour pouvoir chercher dedans vite.
2. **Retrieving** : pour une question, retrouver les passages les plus pertinents.
3. **Augmenting** : mettre ces passages dans le prompt du modèle.
4. **Generating** : le modèle lit ces passages et rédige la réponse.

```
data/raw/vllm-0.10.1 ──► chunks ──► index BM25/TF-IDF ──► top-k passages ──► Qwen3-0.6B ──► réponse
                        (index)       data/processed/     (search)           (answer)
```

---

## 📜 Contraintes du sujet (à respecter absolument)

**Outils imposés**
- Python ≥ 3.10, **uv** comme gestionnaire (le correcteur ne fait que `uv sync`).
- **pydantic** pour les modèles de données échangés entre étapes.
- **Python Fire** pour la CLI, **tqdm** pour les barres de progression.
- Modèle **Qwen/Qwen3-0.6B** (par défaut ; d'autres modèles sont permis si Qwen marche toujours).
- Recherche : **au moins TF-IDF ou BM25** (méthodes « lexicales », basées sur les mots).
- **Deux stratégies de chunking différentes** : une pour le Python, une pour le Markdown/texte.
- Autres librairies : libres.

**Qualité de code**
- flake8 OK, mypy OK, type hints partout, docstrings (PEP 257).
- Aucun crash : `try/except`, context managers (`with open(...)`).
- La CLI est testée avec des entrées pourries : query vide, `k=0`, fichier manquant,
  JSON mal formé… → **jamais de traceback**.
- **Aucun chemin en dur** : tous les chemins passent par des arguments CLI.
- **Interdit** d'importer ou d'appeler la moulinette depuis mon code.

**Performances**
- Indexation du corpus entier : **≤ 5 minutes**.
- Recherche : **≤ 90 secondes pour 200 questions**.
- Recall@5 : **≥ 80 % sur docs**, **≥ 50 % sur code**.
- Chunk max : **2000 caractères** (`--max_chunk_size`, défaut 2000). Une seule source trop
  longue rend **tout** le fichier de sortie invalide.

**Makefile** : `install`, `run`, `debug` (pdb), `clean`, `lint`
(`flake8 .` + `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs`),
`lint-strict` optionnel (`mypy . --strict`).

**Rendu git** : `src/`, `pyproject.toml`, `uv.lock`, `Makefile`, `README.md`, `.gitignore`.
**Pas** de données, de poids de modèle ni de résultats générés dans le repo.

---

## 🗂️ Arborescence imposée

```
.
├── src/                      # mon code, lancé avec: uv run python -m src <commande>
├── pyproject.toml, uv.lock
├── Makefile, README.md (en anglais)
└── data/
    ├── raw/vllm-0.10.1/      # le corpus (à déplacer depuis ./vllm-0.10.1)
    ├── processed/            # l'index produit par `index`
    ├── datasets/
    │   ├── UnansweredQuestions/   # (depuis ./datasets_public/public/)
    │   └── AnsweredQuestions/
    └── output/
        ├── search_results/<UnansweredQuestions|AnsweredQuestions>/
        └── search_results_and_answer/<UnansweredQuestions|AnsweredQuestions>/
```

⚠️ Les `file_path` en sortie doivent être **exactement** `data/raw/vllm-0.10.1/...`
(comparés caractère par caractère).

---

## 🖥️ Commandes CLI obligatoires (Fire)

| Commande | Rôle |
|---|---|
| `index --max_chunk_size 2000` | Lit `data/raw/`, découpe, sauvegarde l'index dans `data/processed/` |
| `search "<query>" --k 10` | Top-k sources pour une seule question |
| `search_dataset --dataset_path P --k 10 --save_directory D` | Recherche sur tout un dataset → JSON `StudentSearchResults` |
| `answer "<query>" --k 10` | Répond à une seule question avec le contexte trouvé |
| `answer_dataset --student_search_results_path P --save_directory D` | Génère les réponses → JSON `StudentSearchResultsAndAnswer` |
| `evaluate --student_search_results_path P --dataset_path P` | Mon propre calcul de recall@k (pour tester) |

Pipeline de la correction :
```bash
uv run python -m src index --max_chunk_size 2000
uv run python -m src search_dataset --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 --save_directory data/output/search_results/UnansweredQuestions
./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json --k 10 --max_context_length 2000
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```
(Renommer `moulinette/moulinette-ubuntu` en `./moulinette` à la racine.)

---

## 🧱 Modèles pydantic (section VI.4 du sujet)

`MinimalSource(file_path, first_character_index, last_character_index)`,
`UnansweredQuestion(question_id = uuid par défaut, question)`,
`AnsweredQuestion(+ sources, answer)`, `RagDataset(rag_questions)`,
`MinimalSearchResults(question_id, question, retrieved_sources)`, `MinimalAnswer(+ answer)`,
`StudentSearchResults(search_results, k)`, `StudentSearchResultsAndAnswer(search_results, k)`.
On peut ajouter des champs / modèles, pas en retirer.

---

## ✅ Comment on est noté (recall@k)

Pour chaque question : le bon passage est-il dans mes k premiers résultats ?
Un résultat compte si c'est **le même fichier** ET que les plages de caractères se
**chevauchent un peu** (IoU ≥ 0.05). Pas besoin de tomber pile sur le passage exact.

Datasets publics : 100 questions docs (sources `.md`), 99 questions code (sources `.py`),
1 source attendue par question, passages de ~800 (code) à ~1100 (docs) caractères en moyenne.

L'évaluation **privilégie la recherche** ; pour les réponses de Qwen (petit modèle limité),
on attend surtout qu'elles soient cohérentes, ancrées dans les sources, et qu'elles répondent à la question.

---

## 🗺️ Feuille de route

Coche au fur et à mesure. Claude : regarde où j'en suis avant de proposer la suite.

### Étape 0 — Mise en place
- [x] `uv init`, dépendances (pydantic, fire, tqdm, flake8, mypy…), `.gitignore`.
- [x] Créer l'arborescence `data/` et y déplacer corpus + datasets.
- [x] `src/__main__.py` + Fire : toutes les commandes existent (même vides).
- [x] `Makefile` complet.

### Étape 1 — Modèles pydantic
- [x] Écrire les modèles du sujet, tester le chargement des JSON de `data/datasets/`.

### Étape 2 — Lire le corpus et chunker
- [x] Parcourir `data/raw/`, garder les fichiers utiles (`.py`, `.md`, `.txt`…).
- [x] Lire **sans modifier le texte**, sinon les index de caractères seront décalés.
- [x] Chunking **Python** (par `def`/`class`, découpé si > max) et chunking **Markdown**
  (par titres `#`, découpé si > max). Chaque chunk retient `file_path`, `start`, `end`, `text`.

### Étape 3 — Index lexical + sauvegarde
- [ ] BM25 (ou TF-IDF) sur les chunks, sauvegardé dans `data/processed/`.
- [ ] Tokenisation adaptée au code : couper `load_lora_adapter` en `load`, `lora`,
  `adapter`, idem pour le camelCase. C'est souvent ce qui fait passer le dataset code.
- [ ] Vérifier : indexation < 5 min.

### Étape 4 — Recherche
- [ ] `search` (1 question) et `search_dataset` (JSON) → sortie `StudentSearchResults`.
- [ ] `evaluate` maison + moulinette officielle. Objectif docs ≥ 80 %, code ≥ 50 %.
- [ ] Itérer **une chose à la fois** (taille de chunk, tokenisation…) et noter les scores.

### Étape 5 — Génération avec Qwen3-0.6B
- [ ] Charger le modèle (transformers, CPU), construire un prompt « contexte + question ».
- [ ] Respecter la limite de tokens du modèle, `answer` et `answer_dataset`.

### Étape 6 — Robustesse + README
- [ ] Tester tous les cas tordus (query vide, k=0, fichiers manquants, JSON cassé).
- [ ] `make lint` propre.
- [x] `.gitignore` : ajouter `data/`, `moulinette/`, `en.subject_RAG.pdf` (reporté par choix ; d'ici là, pas de `git add .`).
- [ ] README **en anglais** : 1re ligne en italique *This project has been created as part of the 42 curriculum by <login>*,
  + Description, Instructions, Resources (+ usage de l'IA), System architecture, Chunking strategy,
  Retrieval method, Performance analysis, Design decisions, Challenges faced, Example usage.

### Bonus (seulement si le mandatory est parfait)
Embeddings (all-MiniLM-L6-v2), recherche hybride, indexation incrémentale, cache, API HTTP locale.

---

## 📝 Suivi de progression

Claude : mets à jour cette section quand une étape est terminée (avec mes scores).

| Date | Étape | Ce qui a été fait | Recall@5 code | Recall@5 docs |
|---|---|---|---|---|
| 2026-10-05 | 0 | Mise en place : uv, arborescence `data/`, CLI Fire (6 commandes vides), Makefile, `.flake8`, `[tool.mypy]`. Reste : types de retour dans `rag_cli.py` (5 erreurs mypy) | – | – |
| 2026-10-07 | 1-2 | Modèles pydantic (`model.py`). Chunking : `CutChunk.chunk_file` (coupe aux `#` pour md/txt, aux `def`/`class` pour py, recoupe à `max_chunk_size`), `ChunkCorpus.chunk_corpus` → 27 248 chunks sur 1 969 fichiers, max 2000 caractères | – | – |
| 2026-10-08 | 3 (en cours) | `Tokenizer.tokenize` (snake_case, camelCase, minuscules). `Bm25Index` dans `src/bm25_index.py` : `__init__` construit `self.bm25 = BM25Okapi(self.tokenize_text())`, test manuel de `get_scores` OK sur 3 chunks | – | – |
| 2026-10-09 | 3 (en cours) | `Bm25Index.list_scores(query)` (zip notes/chunks + tri décroissant) et `Bm25Index.search(query, k)` (k meilleurs chunks). Test manuel OK : `search("How to load LoRA?", 2)` renvoie `load_lora` en premier. flake8 + mypy propres sur `bm25_index.py` | – | – |

**Prochaine étape (où reprendre)** : sauvegarder l'index dans `data/processed/` (commande `index`) et le recharger,
puis mesurer le temps d'indexation sur le corpus entier (limite : 5 min).
À prévoir : `search` avec `k <= 0` ou une question vide (étape 6, pas de traceback).
À revoir à l'étape 4 : la règle camelCase coupe `LoRA` en `lo` + `ra` (ne correspond plus à `lora`).
