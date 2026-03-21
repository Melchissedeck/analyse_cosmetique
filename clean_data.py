import pandas as pd
import numpy as np

print("1. Chargement du dataset brut...")
df = pd.read_parquet("beauty.parquet")

# 2. Sélection des colonnes pertinentes
colonnes_a_garder = [
    'code', 'product_name', 'brands', 'categories_tags', 
    'labels_tags', 'ingredients_tags', 'origins_tags', 
    'scans_n', 'unique_scans_n'
]
df_clean = df[colonnes_a_garder].copy()

print("2. Nettoyage des données de base...")
# Supprimer les lignes où le nom du produit OU la marque est manquant (indispensable pour notre analyse)
df_clean = df_clean.dropna(subset=['product_name', 'brands'])

# Mettre les noms de produits et marques en minuscules pour éviter les doublons (ex: "L'Oreal" vs "l'oreal")
df_clean['brands'] = df_clean['brands'].astype(str).str.lower().str.strip()
df_clean['product_name'] = df_clean['product_name'].astype(str).str.strip()

print("3. Création du proxy de popularité (pour estimer les ventes)...")
# On remplace les valeurs manquantes des scans par 0
df_clean['scans_n'] = df_clean['scans_n'].fillna(0)
df_clean['unique_scans_n'] = df_clean['unique_scans_n'].fillna(0)
# On crée un score de popularité basé sur les scans
df_clean['popularity_score'] = df_clean['scans_n'] + df_clean['unique_scans_n']

print("4. Identification des produits naturels...")
# Une fonction pour vérifier si un label naturel/bio est présent dans la liste des tags
def is_natural(tags):
    if not isinstance(tags, np.ndarray) and not isinstance(tags, list):
        return False
    natural_keywords = ['bio', 'organic', 'natural', 'naturel', 'ecocert', 'cosmos']
    # On convertit les tags en minuscules et on cherche une correspondance
    tags_str = " ".join([str(tag).lower() for tag in tags])
    return any(keyword in tags_str for keyword in natural_keywords)

df_clean['is_natural'] = df_clean['labels_tags'].apply(is_natural)

print("5. Nettoyage des listes (ingrédients, origines, catégories)...")
# Convertir les NaN des listes en listes vides pour éviter les erreurs plus tard
for col in ['categories_tags', 'ingredients_tags', 'origins_tags']:
    df_clean[col] = df_clean[col].apply(lambda x: x if isinstance(x, (list, np.ndarray)) else [])

print("\n--- RÉSUMÉ DU NETTOYAGE ---")
print(f"Nombre de produits restants : {df_clean.shape[0]}")
print(f"Nombre de produits identifiés comme 'Naturels' : {df_clean['is_natural'].sum()}")

# 6. Sauvegarde du dataset nettoyé
print("\n6. Sauvegarde dans 'beauty_cleaned.parquet'...")
df_clean.to_parquet("beauty_cleaned.parquet", engine='pyarrow')
print("Terminé ! Le fichier est prêt pour l'analyse.")