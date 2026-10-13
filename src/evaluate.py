from src.model import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)


class Evaluator:
    """classe qui servira a creer une moulinette.
    Compare les resultats de recherche avec le corriger et
    calcul le pourcentage de reussite (recall@k)."""

    def iou(self, a: MinimalSource, b: MinimalSource) -> float:
        """Mesure le chevauchement en deux passages d'un meme fichier
        (IOU = parite commune / passage total)
        args:
            - a -> passage du corrige
            - b -> passage du resultat trouver par la recherche
        retour attendu entre 0.00 et 1.00 (0% et 100%)."""

        common: int = min(
            a.last_character_index, b.last_character_index
        ) - max(a.first_character_index, b.first_character_index)
        if common <= 0:
            return 0.00
        total: int = max(a.last_character_index, b.last_character_index) - min(
            a.first_character_index, b.first_character_index
        )
        iou: float = common / total
        return iou

    def is_match(self, a: MinimalSource, b: MinimalSource) -> bool:
        """verifie si le bon chemin du fichier et si l'iou est
        superieur a 0.05
            args:
                - a -> corriger
                - b -> resultat generer
        retourne True si les chemins sont identique et que l'iou >= 0.05."""

        iou = self.iou(a, b)
        if a.file_path != b.file_path or iou < 0.05:
            return False
        return True

    def is_in_top_k(
        self, corriger: MinimalSource, results: list[MinimalSource], k: int
    ) -> bool:
        """Compare k resultat au corriger et retourne True si un des resultats
        valide les contitions contenu dans def is_match(si file_path est
        identique et iou >= 0.05)
            args:
                - corriger -> reponse correcte fournis dans le sujet
                - results -> liste de resultat generer par le projet
                - k -> nombre de resultat a verifier
        retourne True si un elements de la liste concorde avec le corrige.
        """
        for _, result in enumerate(results[:k]):
            if self.is_match(corriger, result):
                return True
        return False

    def recall_question(
        self,
        corriges: list[MinimalSource],
        results: list[MinimalSource],
        k: int,
    ) -> float:
        """compte les corriges trouver et divise le nombre d'element de la
        liste corriges pour optenir une note entre 0 et 1 (i = 100% valide)
            args:
                - corriges -> liste de reponse fournis dans le sujet
                - results -> liste de resusltat generer par la recherche
                - k -> nombre d'element a verifier
        retourne une note entre 0 et 1."""

        if not corriges:
            return 0.00
        ok: int = 0
        for corrige in corriges:
            if self.is_in_top_k(corrige, results, k):
                ok += 1
        return ok / len(corriges)

    def recall_dataset(
        self, dataset: RagDataset, student: StudentSearchResults, k: int
    ) -> float:
        """Calculer le score final, le recall@k sur toute les questions.
        C'est ce que la moulinette affiche.
            args:
                - dataset -> le corriger, ragdataset(toutes les questions avec
                les sources)
                - student -> resultats des generer, StudentSearchResults
                (toutes les questions avec les retrivied_source)
                - k -> nombre de resultat qu'on regarde
        retourne un float entre 0 et 1, la moyenne des notes de toutes
        les questions
        """

        results_by_id: dict[str, list[MinimalSource]] = {}

        for search_result in student.search_results:
            results_by_id[search_result.question_id] = (
                search_result.retrieved_sources
            )
        total: float = 0.00
        for question in dataset.rag_questions:
            if not isinstance(question, AnsweredQuestion):
                continue
            results: list[MinimalSource] = results_by_id.get(
                question.question_id, []
            )
            total += self.recall_question(question.sources, results, k)
        return total / len(dataset.rag_questions)
