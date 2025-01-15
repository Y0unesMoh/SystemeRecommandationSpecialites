import pandas as pd

# Charger le fichier CSV
file_path = "donnees_test_independant.csv"  # Remplacez par le chemin de votre fichier
etudiants = pd.read_csv(file_path)

# Définir les spécialités en fonction des filières et des matières principales
filiere_specialites = {
    "Informatique": {
        "python": "Développement Python",
        "bigdata": "Big Data",
        "management_projet": "Gestion de Projet"
    },
    "Cybersécurité": {
        "cloud_virtualisation": "Cloud Computing",
        "anglais": "Analyse de Données",
        "français": "Sécurité Réseau"
    }
}

# Ajouter une colonne pour les spécialités proposées
def proposer_specialite(row):
    filiere = row['filiere']
    preference = row['preference']
    if filiere in filiere_specialites:
        specialites = filiere_specialites[filiere]
        if preference in specialites:
            return specialites[preference]
        else:
            # Choisir la spécialité selon le meilleur score
            matiere_meilleure = max(specialites, key=lambda matiere: row[matiere])
            return specialites[matiere_meilleure]
    return "Non défini"

etudiants['specialite_proposee'] = etudiants.apply(proposer_specialite, axis=1)

# Sauvegarder dans un nouveau fichier CSV
output_file = "C:\\Users\\ASUS\\OneDrive\\Bureau\\Projet0\\etudiants_specialites.csv"
etudiants.to_csv(output_file, index=False)

print(f"Fichier généré : {output_file}")
