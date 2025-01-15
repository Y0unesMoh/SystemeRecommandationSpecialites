import pandas as pd
from surprise import Reader, Dataset, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import random

# Charger les données
df = pd.read_csv('C:\\Users\\ASUS\\OneDrive\\Bureau\\Projet0\\donnees_test_independant.csv')

# Vérification des colonnes nécessaires
expected_columns = ['nom', 'python', 'bigdata', 'anglais', 'cloud_virtualisation', 'management_projet', 'français']
if not all(col in df.columns for col in expected_columns):
    raise ValueError(f"Le fichier CSV doit contenir les colonnes suivantes : {expected_columns}")

# Suppression des doublons et valeurs manquantes
df = df.dropna().drop_duplicates()

# Vérification de l'unicité des noms
if df['nom'].duplicated().any():
    raise ValueError("La colonne 'nom' contient des doublons. Chaque étudiant doit avoir un nom unique.")

# Transformation des scores en format adapté pour Surprise
data = []
for _, row in df.iterrows():
    for col in ['python', 'bigdata', 'anglais', 'cloud_virtualisation', 'management_projet', 'français']:
        data.append([row['nom'], col, row[col]])

# Charger les données dans un format adapté à Surprise
reader = Reader(rating_scale=(0, 100))
data = Dataset.load_from_df(pd.DataFrame(data, columns=['nom', 'specialite', 'score']), reader)

# Diviser les données en train et test
trainset, testset = train_test_split(data, test_size=0.2)

# Apprentissage avec un modèle SVD (Singular Value Decomposition)
algo = SVD()
algo.fit(trainset)

# Tester la précision du modèle
predictions = algo.test(testset)
print("RMSE:", accuracy.rmse(predictions))

# Fonction de recommandation avec Surprise
def recommander_specialites_surprise(student_name, n_top=3):
    # Vérifier si l'étudiant existe dans les données
    if student_name not in df['nom'].values:
        raise ValueError(f"L'étudiant {student_name} n'existe pas dans les données.")
    
    # Trouver les scores des spécialités pour l'étudiant
    specialites = ['python', 'bigdata', 'anglais', 'cloud_virtualisation', 'management_projet', 'français']
    
    recommendations = []
    for specialite in specialites:
        # Prédiction pour l'étudiant
        prediction = algo.predict(student_name, specialite)
        recommendations.append((specialite, prediction.est))
    
    # Trier les recommandations par score décroissant
    recommendations.sort(key=lambda x: x[1], reverse=True)
    
    # Sélectionner les n meilleures spécialités
    top_recommendations = recommendations[:n_top]
    
    return top_recommendations

# Tester avec un étudiant
etudiant = 'Douaa'  # Assurez-vous que cet étudiant existe dans la liste des étudiants
try:
    recommandations = recommander_specialites_surprise(etudiant)
    print(f"Recommandations pour {etudiant} :")
    for rank, (specialite, score) in enumerate(recommandations, start=1):
        print(f"{rank}. {specialite} ({score:.2f} score)")
except ValueError as e:
    print(e)
