import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Définition des données académiques des étudiants
data = {
    'nom': ["Yanis", "Sara", "Ahmed", "Lina", "Zakaria", "Mouhcine", "Rachid", "Nadia", "Imane", "Omar", 
            "Mounir", "Sofia", "Hassan", "Leila", "Samir"],
    'prenom': ["Salhi", "Imane", "Khalid", "Amal", "Omar", "Sofia", "Karim", "Mekouar", "Youssef", "Hassan", 
               "Ali", "Selma", "Yassir", "Naima", "Karim"],
    'age': [23, 21, 22, 20, 24, 21, 23, 22, 19, 25, 22, 23, 24, 20, 21],
    'filiere': ["Informatique", "Informatique", "Cybersécurité", "Cybersécurité", "Informatique", "Informatique",
                "Cybersécurité", "Cybersécurité", "Informatique", "Cybersécurité", "Informatique", "Cybersécurité", 
                "Cybersécurité", "Informatique", "Informatique"],
    'score_matiere_1': [100, 75, 88, 79, 85, 90, 80, 84, 84, 87, 92, 85, 78, 88, 80],
    'score_matiere_2': [85, 82, 90, 72, 80, 83, 84, 92, 75, 85, 78, 95, 87, 79, 84],
    'score_matiere_3': [78, 84, 81, 98, 88, 85, 77, 78, 88, 90, 90, 88, 92, 95, 89],
    'score_matiere_4': [88, 80, 87, 82, 95, 88, 89, 82, 84, 82, 80, 92, 88, 86, 90],
    'score_matiere_5': [80, 88, 83, 82, 80, 86, 85, 70, 90, 91, 87, 85, 84, 88, 83],
    'score_matiere_6': [80, 79, 85, 84, 76, 76, 97, 85, 77, 83, 88, 81, 80, 79, 82],
    'preference': ["python", "management_projet", "bigdata", "anglais", "cloud_virtualisation", "python", 
                   "français", "bigdata", "management_projet", "cloud_virtualisation", "anglais", "bigdata", 
                   "python", "cloud_virtualisation", "anglais"],
    'specialite_proposee': ["Développement Python", "Gestion de Projet", "Big Data", "Langue Anglaise", 
                            "Cloud Computing", "Développement Python", "Langue Française", "Big Data", 
                            "Gestion de Projet", "Gestion de Projet", "Développement Python", "Big Data", 
                            "Langue Anglaise", "Langue Anglaise", "Cloud Computing"]
}

# Conversion des données en DataFrame
df = pd.DataFrame(data)

# Normalisation des scores
scaler = MinMaxScaler()
score_columns = ['score_matiere_1', 'score_matiere_2', 'score_matiere_3', 
                 'score_matiere_4', 'score_matiere_5', 'score_matiere_6']
df[score_columns] = scaler.fit_transform(df[score_columns])

# Création de nouvelles variables
df['score_moyen'] = df[score_columns].mean(axis=1)
df['tendance'] = df['score_matiere_1'] - df['score_matiere_2'] - df['score_matiere_3'] - df['score_matiere_4'] - df['score_matiere_5'] - df['score_matiere_6']

# Déterminer le nombre optimal de clusters (méthode du coude)
inertia = []
for k in range(1, 6):  # Tester de 1 à 5 clusters
    kmeans = KMeans(n_clusters=k, random_state=42)
    kmeans.fit(df[['score_matiere_1', 'score_matiere_2', 'score_matiere_3', 'score_matiere_4', 'score_matiere_5', 'score_matiere_6', 'score_moyen', 'tendance']])
    inertia.append(kmeans.inertia_)

# Tracer la méthode du coude
plt.plot(range(1, 6), inertia, marker='o')
plt.title('Méthode du coude')
plt.xlabel('Nombre de clusters')
plt.ylabel('Inertie')
plt.show()

# Choisir le nombre de clusters (exemple ici avec 3)
optimal_k = 3
kmeans = KMeans(n_clusters=optimal_k, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['score_matiere_1', 'score_matiere_2', 'score_matiere_3', 'score_matiere_4', 'score_matiere_5', 'score_matiere_6', 'score_moyen', 'tendance']])

# Évaluer la qualité des clusters avec le silhouette score
silhouette_avg = silhouette_score(df[['score_matiere_1', 'score_matiere_2', 'score_matiere_3', 'score_matiere_4', 'score_matiere_5', 'score_matiere_6', 'score_moyen', 'tendance']], df['cluster'])
print(f"Silhouette score: {silhouette_avg:.2f}")

# Visualisation des clusters (en 3D)
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.scatter(df['score_matiere_1'], df['score_matiere_2'], df['score_matiere_3'], c=df['cluster'], cmap='viridis')
ax.set_xlabel('Score Matière 1 (Normalisé)')
ax.set_ylabel('Score Matière 2 (Normalisé)')
ax.set_zlabel('Score Matière 3 (Normalisé)')
plt.title("Clusters d'étudiants")
plt.show()

# Affichage du DataFrame final
print(df)
