import pandas as pd
import numpy as np
import re

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
# Supprimer les NaN
df_clean = df_clean.dropna(subset=['product_name', 'brands'])

# Mettre en minuscules et retirer les espaces inutiles
df_clean['brands'] = df_clean['brands'].astype(str).str.lower().str.strip()

# Fonction pour extraire le vrai nom du produit de la structure complexe
def extraire_nom_produit(valeur):
    valeur_str = str(valeur).strip()
    
    # Si la valeur est vide, nulle, ou une liste vide
    if valeur_str in ["[]", "nan", "None", ""]:
        return "Nom inconnu"
        
    # Regex pour capturer ce qui est entre guillemets après 'text':
    match = re.search(r"'text':\s*(['\"])(.*?)\1", valeur_str)
    if match:
        return match.group(2).strip()
        
    return valeur_str

# Application de la fonction
df_clean['product_name'] = df_clean['product_name'].apply(extraire_nom_produit)

# supprimer les produits dont le nom est resté "Nom inconnu"
df_clean = df_clean[df_clean['product_name'] != "Nom inconnu"]

# Supprimer les marques vides (qui n'étaient pas des NaN)
df_clean = df_clean[df_clean['brands'] != ""]

# Gérer les marques multiples (séparées par des virgules)
# On ne garde que la première entité (qui est généralement la marque mère)
df_clean['brands'] = df_clean['brands'].apply(lambda x: x.split(',')[0].strip())

# Unifier les grands groupes (Regroupement sémantique)
def unifier_marques(marque):
    if "l'oréal" in marque or "l'oreal" in marque or "loreal" in marque:
        return "l'oréal"
    if "unilever" in marque:
        return "unilever"
    if "nivea" in marque:
        # Nivea appartient à Beiersdorf
        return "nivea (beiersdorf)"
    if "garnier" in marque or "mixa" in marque or "cadum" in marque:
        return "l'oréal"
    if "dove" in marque or "axe" in marque or "sanex" in marque:
        return "unilever"
    return marque

df_clean['brands'] = df_clean['brands'].apply(unifier_marques)

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