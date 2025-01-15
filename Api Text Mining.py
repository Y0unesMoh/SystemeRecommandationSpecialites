import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk import pos_tag, ne_chunk
from nltk.tree import Tree

# Assurez-vous de télécharger les données nécessaires pour nltk
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('maxent_ne_chunker')
nltk.download('words')

# Charger le fichier CSV
file_path = 'donnees_test_independant.csv'  # Chemin du fichier CSV
df = pd.read_csv(file_path)

# Définir la colonne contenant les textes à analyser
text_column = 'specialite_proposee'  # Nom de la colonne à analyser

# Fonction pour extraire les entités nommées avec nltk
def extract_entities(text):
    try:
        # Tokenisation et étiquetage des parties du discours
        tokens = word_tokenize(text)
        pos_tags = pos_tag(tokens)
        
        # Analyse des entités nommées
        chunked = ne_chunk(pos_tags)
        entities = []
        
        # Parcours des arbres pour extraire les entités nommées
        for subtree in chunked:
            if isinstance(subtree, Tree):  # Si c'est une entité nommée
                entity = " ".join([token for token, pos in subtree.leaves()])
                entities.append(entity)
        
        return entities
    except Exception as e:
        print(f"Erreur lors de l'analyse du texte : {text} - {e}")
        return []

# Analyser chaque ligne du DataFrame
results = []

for index, row in df.iterrows():
    texte = row[text_column]
    entities = extract_entities(texte)
    results.append({
        'texte': texte,
        'entities': ', '.join(entities)  # Joindre les entités
    })

# Convertir les résultats en DataFrame
results_df = pd.DataFrame(results)

# Afficher un aperçu des résultats
print(results_df.head())

# Sauvegarder les résultats dans un nouveau fichier CSV
output_file_path = 'analyse_texte_nltk_resultats.csv'
results_df.to_csv(output_file_path, index=False)

print(f"Les résultats ont été sauvegardés avec succès dans : {output_file_path}")
