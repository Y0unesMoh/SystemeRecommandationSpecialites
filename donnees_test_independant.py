import csv

# Données des étudiants
data = {
    "nom": ["Yanis", "Sara", "Ahmed", "Lina", "Zakaria", "Mouhcine", "Rachid", "Nadia", "Imane", "Omar",
            "Mounir", "Sofia", "Hassan", "Leila", "Samir"],
    "prenom": ["Salhi", "Imane", "Khalid", "Amal", "Omar", "Sofia", "Karim", "Mekouar", "Youssef", "Hassan",
               "Ali", "Selma", "Yassir", "Naima", "Karim"],
    "age": [23, 21, 22, 20, 24, 21, 23, 22, 19, 25,
            22, 23, 24, 20, 21],
    "filiere": ["Informatique", "Informatique", "Cybersécurité", "Cybersécurité", "Informatique",
                "Informatique", "Cybersécurité", "Cybersécurité", "Informatique", "Cybersécurité",
                "Informatique", "Cybersécurité", "Cybersécurité", "Informatique", "Informatique"],
    "python": [100, 75, 88, 79, 85, 90, 80, 84, 84, 87,
               92, 85, 78, 88, 80],
    "bigdata": [85, 82, 90, 72, 80, 83, 84, 92, 75, 85,
                78, 95, 87, 79, 84],
    "anglais": [78, 84, 81, 98, 88, 85, 77, 78, 88, 90,
                90, 88, 92, 95, 89],
    "cloud_virtualisation": [88, 80, 87, 82, 95, 88, 89, 82, 84, 82,
                             80, 92, 88, 86, 90],
    "management_projet": [80, 88, 83, 82, 80, 86, 85, 70, 90, 91,
                          87, 85, 84, 88, 83],
    "français": [80, 79, 85, 84, 76, 76, 97, 85, 77, 83,
                 88, 81, 80, 79, 82],
                 
    "preference": ["python", "management_projet", "bigdata", "anglais", "cloud_virtualisation", "python",
                   "français", "bigdata", "management_projet", "cloud_virtualisation",
                   "anglais", "bigdata", "python", "cloud_virtualisation", "anglais"]
}

# Spécialités correspondantes aux matières
specialites = {
    "python": "Développement Python",
    "bigdata": "Big Data",
    "anglais": "Langue Anglaise",
    "cloud_virtualisation": "Cloud Computing",
    "management_projet": "Gestion de Projet",
    "français": "Langue Française"
}

# Fonction pour attribuer la spécialité en fonction de la meilleure note
def attribuer_specialite(data):
    specialite_finale = []
    for i in range(len(data["nom"])):
        # Créez un dictionnaire des matières et des notes de l'étudiant
        notes = {
            "python": data["python"][i],
            "bigdata": data["bigdata"][i],
            "anglais": data["anglais"][i],
            "cloud_virtualisation": data["cloud_virtualisation"][i],
            "management_projet": data["management_projet"][i],
            "français": data["français"][i]
        }
        # Identifier la matière avec la meilleure note
        meilleure_matiere = max(notes, key=notes.get)
        # Associer la spécialité correspondante
        specialite_finale.append(specialites[meilleure_matiere])
    return specialite_finale

# Mise à jour des spécialités
data["specialite_proposee"] = attribuer_specialite(data)

# Préparation des données pour l'exportation CSV
header = ["nom", "prenom", "age", "filiere", "python", "bigdata", "anglais", "cloud_virtualisation", 
          "management_projet", "français", "preference", "specialite_proposee"]

rows = []
for i in range(len(data["nom"])):
    rows.append([data["nom"][i], data["prenom"][i], data["age"][i], data["filiere"][i], 
                 data["python"][i], data["bigdata"][i], data["anglais"][i], 
                 data["cloud_virtualisation"][i], data["management_projet"][i], 
                 data["français"][i], data["preference"][i], data["specialite_proposee"][i]])

# Enregistrement dans un fichier CSV
file_path = 'C:\\Users\\ASUS\\OneDrive\\Bureau\\Projet0\\donnees_test_independant.csv'
with open(file_path, mode='w', newline='', encoding='utf-8') as file:
    writer = csv.writer(file)
    writer.writerow(header)
    writer.writerows(rows)

file_path