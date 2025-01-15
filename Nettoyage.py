import pandas as pd
from sklearn.preprocessing import MinMaxScaler

# Charger les données depuis le fichier CSV
file_path = "C:\\Users\\ASUS\\OneDrive\\Bureau\\Projet0\\donnees_test_independant.csv"
df = pd.read_csv(file_path)

# Vérification des colonnes nécessaires
expected_columns = ['python', 'bigdata', 'anglais', 'cloud_virtualisation', 'management_projet', 'français']

if not all(col in df.columns for col in expected_columns):
    raise ValueError(f"Le fichier CSV doit contenir les colonnes suivantes : {expected_columns}")

# Suppression des valeurs manquantes et des doublons
df = df.dropna()
df = df.drop_duplicates()

# Normalisation des scores entre 0 et 100
scaler = MinMaxScaler(feature_range=(0, 100))
df[['python', 'bigdata', 'anglais', 'cloud_virtualisation', 'management_projet', 'français']] = scaler.fit_transform(
    df[['python', 'bigdata', 'anglais', 'cloud_virtualisation', 'management_projet', 'français']]
)

# Afficher les premières lignes et le type des colonnes pour vérification
print(df.head(50))
