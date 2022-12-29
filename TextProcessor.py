import spacy
import csv
from langdetect import detect
from collections import Counter

# Chargement des modèles spaCy
nlp_en = spacy.load("en_core_web_sm")
nlp_fr = spacy.load("fr_core_news_sm")


# Définir la classe TextProcessor
class TextProcessor:
    """
    Dans la méthode d'initialisation de la classe,
    on définit le nom du fichier et les processeurs de texte,
    et on initialise les variables d'instance.
    """
    def __init__(self, filename):
        self.filename = filename  # Nom du fichier
        self.nlp_en = nlp_en  # Processeur de texte en anglais
        self.nlp_fr = nlp_fr  # Processeur de texte en français
        self.tokens = []  # Liste de token
        self.lemmas = []  # Liste de lemme
        self.pos = []  # Liste de POS tagging
        self.text = None  # Initialisation de la variable d'instance text à None

    # Définir la méthode de chargement du texte
    def load_text(self):
        # Ouvrir le fichier et lire toutes les lignes
        with open(self.filename, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            # Si le nombre de lignes de texte n'est pas compris entre 5 et 100, afficher un message d'erreur.
            if len(lines) < 5 or len(lines) > 100:
                print(f"The file {self.filename} contains {len(lines)} lines, does not meet the requirements,"
                      f"the text length needs to be between 5-100 lines.")
                return  # Arrêter l'exécution du code suivant si la longueur du texte ne répond pas aux exigences.
            else:
                print(f"The file {self.filename} has been loaded successfully.")

        # Stocker les lignes de texte dans la variable d'instance text
        self.text = lines
        return lines  # Si le chargement est réussi, on renvoie toutes les lignes de texte.

    # Définir la représentation des chaînes de caractères de la classe
    def __repr__(self):
        return f"Text file: {self.filename}"

    def spacy_process(self):
        # Concaténer la liste des chaînes de caractères en une seule chaîne.
        text = '\n'.join(self.text)
        # Détecter la langue du texte
        language = detect(text)
        print(f"The file {self.filename} is in {language}.")
        return language

    # Définir la méthode de traitement de texte à l'aide de spaCy
    def spacy(self):
        language = self.spacy_process()

        # Déterminer quel processeur de texte utiliser en fonction de la langue détectée
        if language == "en":
            nlp = self.nlp_en
        elif language == "fr":
            nlp = self.nlp_fr
        else:
            raise ValueError(f"Unsupported language: {language}")

        # Concaténer la liste des chaînes de caractères en une seule chaîne.
        text = '\n'.join(self.text)
        # Traiter le texte avec SpaCy
        doc = nlp(text)

        # Itérer sur chaque token dans le texte traité
        for token in doc:
            self.tokens.append(token.text)  # Tokenisation et l'ajouter dans la liste
            self.lemmas.append(token.lemma_)  # Lemmatisation et l'ajouter dans la liste
            self.pos.append(token.pos_)  # POS tagging et l'ajouter dans la liste
        # À la fin de la boucle, il ajoute les valeurs dans leurs listes.
        return self.tokens, self.lemmas, self.pos

    # Définir la méthode de fréquence statistique des tokens
    def word_counts(self):
        word_counts = Counter(self.tokens)
        return word_counts

    # Définir la méthode pour écrire les résultats ci-dessus dans un fichier CSV
    def write_csv(self, filename):
        # Calculer la fréquence de mots
        word_counts = self.word_counts()

        #Ouvrir le nom de fichier spécifié dans l'argument pour l'écriture et crée un objet fichier csv.
        with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
            # Définir les noms des champs qui seront utilisés dans le fichier et crée un objet csv.DictWriter.
            fieldnames = ['index', 'Token', 'Lemma', 'POS', 'word_count']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader() # Écrire le
            # itérer sur les listes et écrire une ligne dans le fichier pour chaque élément de ces listes.
            for index, (token, lemma, pos) in enumerate(zip(self.tokens, self.lemmas, self.pos)):
                # La fonction enumerate renvoie un tuple de l'index de l'élément et de l'élément lui-même.
                # La fonction zip combine les trois listes en un itérable de tuples
                writer.writerow(
                    {'index': index + 1, 'Token': token, 'Lemma': lemma, 'POS': pos, 'word_count': word_counts[token]})
            print(f"Results saved in {filename}")


# Créer un objet de la classe TextProcessor en lui passant le nom du fichier de texte à traiter
text_processor = TextProcessor("text_en.txt")
# Charger le texte à partir du fichier
lines = text_processor.load_text()
# Traiter le texte avec spaCy et obtenir les tokens, lemmes et pos tags
tokens, lemmas, pos = text_processor.spacy()
# Obtenir les fréquences statistiques des tokens
word_counts = text_processor.word_counts()
# Écrire les résultats dans un fichier CSV
text_processor.write_csv("output_test_en.csv")
