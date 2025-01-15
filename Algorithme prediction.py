import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Chargement des données
file_path = "C:\\Users\\ASUS\\OneDrive\\Bureau\\Projet0\\donnees_test_independant.csv"
try:
    df = pd.read_csv(file_path)
except FileNotFoundError:
    raise FileNotFoundError(f"Le fichier spécifié est introuvable : {file_path}")

# Prétraitement des données
df = df.dropna()  # Suppression des valeurs manquantes
df = df.drop_duplicates()  # Suppression des doublons

# Liste des colonnes des scores
scores = ['python', 'bigdata', 'anglais', 'cloud_virtualisation', 
          'management_projet', 'français']

# Normalisation des scores
scaler = MinMaxScaler(feature_range=(0, 100))
df[scores] = scaler.fit_transform(df[scores])

# Définition des variables X (features) et y (cible)
X = df[scores]  # Utilisation des colonnes de scores comme features
y = df['specialite_proposee']  # Colonne cible

# Encodage des labels pour la colonne 'Spécialité'
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Division des données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Entraînement du modèle Random Forest
model = RandomForestClassifier(
    n_estimators=100,  # Augmentation du nombre d'arbres pour améliorer la précision
    random_state=50,
    class_weight='balanced',  # Gestion des classes déséquilibrées
    max_depth=10  # Limitation de la profondeur pour éviter le surapprentissage
)
model.fit(X_train, y_train)

# Récupérer les classes effectivement présentes dans les ensembles d'entraînement/test
classes_presentes = sorted(set(y_train) | set(y_test))
noms_classes = le.inverse_transform(classes_presentes)  # Classes correspondantes en texte

# Évaluation du modèle
y_pred = model.predict(X_test)
print("\nRapport de classification :")
print(classification_report(
    y_test, y_pred, 
    target_names=noms_classes, 
    labels=classes_presentes, 
    zero_division=1
))
print("\nMatrice de confusion :")
print(confusion_matrix(y_test, y_pred, labels=classes_presentes))

# Test avec un nouvel étudiant
nouvel_etudiant = pd.DataFrame([[90, 85, 88, 92, 84, 83]], columns=scores)

# Normalisation des scores du nouvel étudiant
nouvel_etudiant[scores] = scaler.transform(nouvel_etudiant[scores])

# Prédiction pour le nouvel étudiant
prediction = model.predict(nouvel_etudiant)
specialite_predite = le.inverse_transform(prediction)
print(f"\nLa spécialité prédite pour le nouvel étudiant est : {specialite_predite[0]}")
