import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Charger et normaliser les données
def charger_et_normaliser(file_path, scores_columns):
    df = pd.read_csv(file_path)
    if df.empty:
        raise ValueError("Le fichier CSV est vide ou mal chargé.")
    
    # Normalisation des scores
    scaler = MinMaxScaler()
    scores_normalized = scaler.fit_transform(df[scores_columns])
    df[scores_columns] = pd.DataFrame(scores_normalized, columns=scores_columns)
    
    # Calcul du score moyen
    df['score_moyen'] = df[scores_columns].mean(axis=1)
    return df

# Analyser la matière préférée d'un étudiant
def analyser_matiere_preference(df, student_name, scores_columns):
    if student_name not in df['nom'].values:
        return f"L'étudiant {student_name} n'existe pas dans les données."
    
    # Extraire les scores de l'étudiant
    scores_etudiant = df.loc[df['nom'] == student_name, scores_columns].values.flatten()
    
    # Trouver la matière avec le score maximal
    matiere_preferee = scores_columns[scores_etudiant.argmax()]
    return matiere_preferee

# Recommander par contenu
def recommander_par_contenu(df, student_name, scores_columns):
    if student_name not in df['nom'].values:
        return f"L'étudiant {student_name} n'existe pas dans les données."
    
    # Extraire les scores de l'étudiant
    scores_etudiant = df.loc[df['nom'] == student_name, scores_columns].values.flatten()
    
    # Calcul de la similarité
    similarite = df[scores_columns].dot(scores_etudiant)
    df['similarite'] = similarite
    
    # Retourner les spécialités avec la plus forte similarité
    return df.groupby('specialite_proposee')['similarite'].mean().sort_values(ascending=False).head(3)

# Recommander par matière
def recommander_par_matiere(df, student_name, scores_columns):
    matiere_specialites = {
        'cloud_virtualisation': 'Cloud Computing',
        'python': 'Développement Python',
        'anglais': 'Langue Anglaise',
        'management_projet': 'Gestion de Projet',
        'bigdata': 'Big Data',
        'français': 'Langue francaise',
    }

    # Trouver la matière préférée
    matiere_preferee = analyser_matiere_preference(df, student_name, scores_columns)
    return matiere_specialites.get(matiere_preferee, "La spécialité varie en fonction des intérêts")

# Exemple d'utilisation
file_path = 'C:\\Users\\ASUS\\OneDrive\\Bureau\\Projet0\\donnees_test_independant.csv'
# Corriger la liste des colonnes
scores_columns = ['python', 'bigdata', 'anglais', 'cloud_virtualisation', 
          'management_projet', 'français']

# Charger et normaliser les données
df = charger_et_normaliser(file_path, scores_columns)

# Exemple d'étudiant
etudiant_name = 'Yanis'

# Recommandations basées sur le contenu et sur la matière
recommandations_contenu = recommander_par_contenu(df, etudiant_name, scores_columns)
recommandations_matiere = recommander_par_matiere(df, etudiant_name, scores_columns)

# Affichage des résultats
print(f"Recommandations basées sur le contenu pour {etudiant_name} :\n{recommandations_contenu}")
print(f"Recommandation basée sur la matière préférée : {recommandations_matiere}")
