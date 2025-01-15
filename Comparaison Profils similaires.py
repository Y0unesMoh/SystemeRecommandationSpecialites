import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics.pairwise import cosine_similarity

# Données des étudiants avec les nouveaux scores
data = {
    'nom': ['Yanis', 'Sara', 'Ahmed', 'Lina', 'Zakaria', 'Mouhcine', 'Rachid', 'Nadia', 'Imane', 'Omar', 'Mounir', 'Sofia', 
            'Hassan', 'Leila', 'Samir'],
    'python': [100, 75, 88, 79, 85, 90, 80, 84, 84, 87, 92, 85, 78, 88, 80],
    'bigdata': [85, 82, 90, 72, 80, 83, 84, 92, 75, 85, 78, 95, 87, 79, 84],
    'anglais': [78, 84, 81, 98, 88, 85, 77, 78, 88, 90, 90, 88, 92, 95, 89],
    'cloud_virtualisation': [88, 80, 87, 82, 95, 88, 89, 82, 84, 82, 80, 92, 88, 86, 90],
    'management_projet': [80, 88, 83, 82, 80, 86, 97, 70, 77, 91, 87, 92, 80, 88, 83],
    'français': [80, 79, 85, 84, 76, 76, 86, 85, 79, 83, 88, 81, 79, 80, 77],
    'specialite': ['Informatique', 'Informatique', 'Cybersécurité', 'Cybersécurité', 'Informatique', 'Informatique', 'Cybersécurité', 'Cybersécurité', 'Informatique', 'Cybersécurité', 
                   'Informatique', 'Cybersécurité', 'Cybersécurité', 'Informatique', 'Informatique']
}

# Créer un DataFrame
df = pd.DataFrame(data)

# Réindexer les lignes pour commencer à 1
df.index = df.index + 1

# Liste correcte des colonnes à normaliser
scores_columns = [
'python', 'bigdata', 'anglais', 'cloud_virtualisation', 'management_projet', 'français'
]

# Normaliser les scores
scaler = MinMaxScaler()
df[scores_columns] = scaler.fit_transform(df[scores_columns])

# Fonction pour trouver l'étudiant le plus similaire
def trouver_etudiant_similaire(etudiant):
    # Vérifier si l'étudiant existe dans le DataFrame
    if etudiant not in df['nom'].values:
        return f"L'étudiant {etudiant} n'existe pas dans les données."
    
    # Extraire les scores de l'étudiant
    profil_etudiant = df.loc[df['nom'] == etudiant, scores_columns].values.flatten()
    
    # Calculer la similarité avec tous les autres étudiants
    similarites = []
    for other_student in df['nom']:
        if other_student != etudiant:
            profil_autre = df.loc[df['nom'] == other_student, scores_columns].values.flatten()
            sim = cosine_similarity([profil_etudiant], [profil_autre])[0][0]
            similarites.append((other_student, sim))
    
    # Trouver l'étudiant le plus similaire
    similarites.sort(key=lambda x: x[1], reverse=True)
    return similarites[0]

# Trouver l'étudiant le plus similaire à un autre
etudiant_similaire = trouver_etudiant_similaire('Yanis')
print(f"L'étudiant le plus similaire à Yanis est : {etudiant_similaire[0]} avec une similarité de {etudiant_similaire[1]:.2f}")
