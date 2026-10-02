class RagCli:
    """class contenant toutes les methodes pour
    exectuter les commande d'entree que fire
    transformera"""

    def index(self, max_chunk_size: int = 2000) -> None:
        """Decoupe les fichiers en paquet de 2000 caractere
        max dans data/raw/vllm-0.10.1 et sauvegarde l'index
        dans data/processed"""

        print(f"max_chunk_size: {max_chunk_size}")

    def search(self, query: str, k: int = 10):
        """permet de retourner les k meilleurs chunk du
        classement.
            args:
                - query -> questions posee
                - k -> nombre de chunk"""

        print(f"query: {query}, k: {k}")

    def search_dataset(
        self, dataset_path: str, save_directory: str, k: int = 10
    ):
        """retourne les k meilleurs chunk d'un fichier de question
        et l'enregistre dans fichier JSON.
            args:
                - dataset_path -> chemin du fichier JSON contenant
                  les questions
                - save_directory -> document est stocker le fichier
                  des reponses
                - k -> nombre de chunk a retourner par question"""

    def answer(self, query: str, k: int = 10):
        """retourne les k meilleurs chunk et l'envois au model Qwen afin de
        rediger une reponse en texte.
            args:
                - query -> question posee
                - k -> nombre de chunk a retourner"""

    def answer_dataset(
        self, student_search_results_path: str, save_directory: str
    ):
        """redige les reponses de tous les un fichier de question et les
        enregistre dans un fichier JSON (elle lis le fichier produit par
        search_dataset).
            args:
                - student_search_results_path -> chemin du fichier JSON
                  produit par la fonction search_dataset
                - save_directory -> document est stocker le fichier des
                  reponses"""

    def evaluate(self, student_search_results_path: str, dataset_path: str):
        """compare les resultats de recherche avec le corriger et creer un
        pourcentage de reussite.
            args:
                - student_search_results_path -> chemin du fichier JSON
                  produit par la fonction search_dataset
                - chemin du fichier JSON contenant
                  les questions"""
