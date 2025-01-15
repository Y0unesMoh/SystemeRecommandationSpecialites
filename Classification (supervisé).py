import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder

# Charger les données
data = {
    "nom": ["Yanis", "Sara", "Ahmed", "Lina", "Zakaria", "Mouhcine", "Rachid", "Nadia", "Imane", "Omar", "Mounir", "Sofia", "Hassan", "Leila", "Samir"],
    "prenom": ["Salhi", "Imane", "Khalid", "Amal", "Omar", "Sofia", "Karim", "Mekouar", "Youssef", "Hassan", "Ali", "Selma", "Yassir", "Naima", "Karim"],
    "age": [23, 21, 22, 20, 24, 21, 23, 22, 19, 25, 22, 23, 24, 20, 21],
    "filiere": ["Informatique", "Informatique", "Cybersécurité", "Cybersécurité", "Informatique", "Informatique", "Cybersécurité", "Cybersécurité", 
                "Informatique", "Cybersécurité", "Informatique", "Cybersécurité", "Cybersécurité", "Informatique", "Informatique"],
    "python": [100, 75, 88, 79, 85, 90, 80, 84, 84, 87, 92, 85, 78, 88, 80],
    "bigdata": [85, 82, 90, 72, 80, 83, 84, 92, 75, 85, 78, 95, 87, 79, 84],
    "anglais": [78, 84, 81, 98, 88, 85, 77, 78, 88, 90, 90, 88, 92, 95, 89],
    "cloud_virtualisation": [88, 80, 87, 82, 95, 88, 89, 82, 84, 82, 80, 92, 88, 86, 90],
    "management_projet": [80, 88, 83, 82, 80, 86, 85, 70, 90, 91, 87, 85, 84, 88, 83],
    "français": [80, 79, 85, 84, 76, 76, 97, 85, 77, 83, 88, 81, 80, 79, 82],
    "preference": ["python", "management_projet", "bigdata", "anglais", "cloud_virtualisation", "python", "français", "bigdata", 
                   "management_projet", "cloud_virtualisation", "anglais", "bigdata", "python", "cloud_virtualisation", "anglais"],
    "specialite_proposee": ["Développement Python", "Gestion de Projet", "Big Data", "Langue Anglaise", "Cloud Computing", "Développement Python", 
                            "Langue Française", "Big Data", "Gestion de Projet", "Gestion de Projet", "Développement Python", "Big Data", 
                            "Langue Anglaise", "Langue Anglaise", "Cloud Computing"]
}

df = pd.DataFrame(data)

# Colonnes des caractéristiques et cible
feature_columns = ['python', 'bigdata', 'anglais', 'cloud_virtualisation', 'management_projet', 'français']
target_column = 'specialite_proposee'

# Vérification des colonnes
if not all(col in df.columns for col in feature_columns + [target_column]):
    raise ValueError("Les colonnes spécifiées pour les caractéristiques ou la cible n'existent pas.")

# Imputation des valeurs manquantes
df[feature_columns] = df[feature_columns].fillna(df[feature_columns].mean())

# Vérification des types de données
if not all(pd.api.types.is_numeric_dtype(df[col]) for col in feature_columns):
    raise ValueError("Certaines colonnes de caractéristiques ne sont pas numériques.")

# Séparation des caractéristiques et de la cible
X = df[feature_columns]
y = df[target_column]

# Encoder les labels (cible) en valeurs numériques
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Séparer les données en ensemble d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Entraîner le modèle
model = RandomForestClassifier(random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

# Prédire sur l'ensemble de test
y_pred = model.predict(X_test)

# Liste des classes pour forcer leur présence dans le rapport
classes = le.classes_

# Évaluation du modèle
print("Rapport de classification :\n", classification_report(y_test, y_pred, target_names=classes, labels=range(len(classes))))
print(f"Précision : {accuracy_score(y_test, y_pred):.2f}")
print("Matrice de confusion :\n", confusion_matrix(y_test, y_pred, labels=range(len(classes))))
