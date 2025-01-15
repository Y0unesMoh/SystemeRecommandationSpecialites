import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import keras_tuner as kt

# Charger les données
df = pd.read_csv("C:\\Users\\ASUS\\OneDrive\\Bureau\\Projet0\\donnees_test_independant.csv")

# Vérification des colonnes nécessaires dans les données
expected_columns = ['python', 'bigdata', 'anglais', 'cloud_virtualisation', 'management_projet', 'français', 'specialite_proposee']
missing_columns = [col for col in expected_columns if col not in df.columns]
if missing_columns:
    raise ValueError(f"Les colonnes suivantes manquent dans les données : {missing_columns}")

# Sélectionner les caractéristiques et la cible
X = df[['python', 'bigdata', 'anglais', 'cloud_virtualisation', 'management_projet', 'français']]  # Sélectionner toutes les caractéristiques
y = df['specialite_proposee']

# Encoder les labels (spécialité)
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Normaliser les caractéristiques
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Diviser les données en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Définir une fonction pour construire le modèle
def build_model(hp):
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(hp.Int('units_1', min_value=32, max_value=128, step=16), activation='relu'),
        tf.keras.layers.Dense(hp.Int('units_2', min_value=16, max_value=64, step=16), activation='relu'),
        tf.keras.layers.Dense(len(le.classes_), activation='softmax')
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Choice('learning_rate', [1e-3, 1e-4, 1e-5])),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# Configurer le tuner
tuner = kt.RandomSearch(
    build_model,
    objective='val_accuracy',
    max_trials=10,
    directory='my_tuner_dir',
    project_name='student_specialty'
)

# Lancer la recherche des hyperparamètres
tuner.search(X_train, y_train, epochs=50, validation_split=0.2, batch_size=16)

# Meilleurs hyperparamètres trouvés
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
model = tuner.hypermodel.build(best_hps)

# Entraîner le modèle final
history = model.fit(X_train, y_train, epochs=50, batch_size=16, validation_data=(X_test, y_test), verbose=2)

# Évaluation sur l'ensemble de test
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f"Précision sur l'ensemble de test : {test_accuracy * 100:.2f}%")

# Simulation des nouveaux étudiants
X_simulated = np.array([
    [100, 85, 78, 88, 80, 80],  # Yanis Salhi
    [75, 82, 84, 80, 90, 79],   # Sara Imane
    [88, 90, 81, 87, 83, 85],   # Ahmed Khalid
    [79, 72, 98, 82, 82, 84],   # Lina Amal
    [85, 80, 88, 95, 80, 76],   # Zakaria Omar
    [90, 83, 85, 88, 86, 76],   # Mouhcine Sofia
    [80, 84, 77, 89, 85, 97],   # Rachid Karim
    [84, 92, 78, 82, 70, 85],   # Nadia Mekouar
    [84, 75, 88, 84, 90, 77],   # Imane Youssef
    [87, 85, 81, 88, 50, 83],   # Omar Hassan
    [92, 78, 89, 80, 87, 88],   # Mounir Ali
    [85, 95, 88, 92, 85, 81],   # Sofia Selma
    [100, 87, 90, 88, 84, 80],  # Hassan Yassir
    [88, 79, 86, 95, 88, 79],   # Leila Naima
    [80, 84, 97, 90, 83, 82]    # Samir Karim
])

# Vérifier les dimensions de X_simulated
if X_simulated.shape[1] != X_train.shape[1]:
    raise ValueError(f"X_simulated doit avoir {X_train.shape[1]} colonnes, mais il en a {X_simulated.shape[1]}.")

# Normaliser les nouveaux scores avec la même normalisation
X_simulated_scaled = scaler.transform(X_simulated)

# Prédictions sur les nouveaux étudiants simulés
predictions_simulees = model.predict(X_simulated_scaled)
specialites_predites_simulees = le.inverse_transform(np.argmax(predictions_simulees, axis=1))

# Afficher les prédictions pour les étudiants simulés
for i, pred in enumerate(specialites_predites_simulees):
    print(f"La spécialité prédite pour l'étudiant {i+1} (scores {X_simulated[i]}): {pred}")

# Sauvegarder le modèle après l'entraînement
model.save("modele_tensorflow_final.keras")
print("Modèle sauvegardé")