import pandas as pd

# 1. Chargement des données
print("Chargement du dataset en cours...")
df = pd.read_parquet("beauty.parquet")

# 2. Aperçu général
print("\n--- APERÇU GÉNÉRAL ---")
print(f"Nombre de lignes (produits) : {df.shape[0]}")
print(f"Nombre de colonnes : {df.shape[1]}")

# 3. Afficher les noms des colonnes pour voir ce qu'on peut utiliser
print("\n--- LISTE DES COLONNES ---")
print(df.columns.tolist())

# 4. Vérifier les valeurs manquantes (pourcentage par colonne)
print("\n--- POURCENTAGE DE VALEURS MANQUANTES ---")
missing_pct = (df.isnull().sum() / len(df)) * 100
# On n'affiche que les colonnes qui ont moins de 100% de valeurs manquantes pour y voir clair
print(missing_pct[missing_pct < 100].sort_values(ascending=False).head(20))

# 5. Afficher les 3 premières lignes
print("\n--- 3 PREMIÈRES LIGNES ---")
print(df.head(3))