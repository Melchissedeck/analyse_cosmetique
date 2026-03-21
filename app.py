import streamlit as st
import pandas as pd
import numpy as np
from collections import Counter
import altair as alt  # <-- NOUVEAU : Pour le graphique en camembert

# 1. Configuration de la page
st.set_page_config(page_title="Analyse Marché Cosmétique", layout="wide")
st.title("💄 Dashboard : Industrie Cosmétique & Clean Beauty")

# 2. Chargement des données nettoyées
@st.cache_data
def load_data():
    return pd.read_parquet("beauty_cleaned.parquet")

df = load_data()

# --- SECTION 1 : MACRO ÉCONOMIE (Chiffres réels) ---
st.header("1. Aperçu Financier Mondial (2025)")
col1, col2, col3 = st.columns(3)
col1.metric("Revenus Globaux", "354.68 Mds $")
col2.metric("Leader Mondial", "L'Oréal (40.3 Mds $)")
col3.metric("Premier Segment", "Soins Capillaires (24%)")

# --- SECTION 2 : TOP ENTREPRISES ET PRODUITS ---
st.header("2. Analyse des Marques et Produits (via Open Beauty Facts)")
col_a, col_b = st.columns(2)

with col_a:
    st.subheader("Top 50 des Entreprises (par nombre de produits)")
    top_brands = df['brands'].value_counts().head(50)
    st.bar_chart(top_brands)

with col_b:
    st.subheader("Produits Naturels les plus populaires")
    
    # Préparation des données naturelles
    df_nat = df[df['is_natural'] == True].sort_values(by='popularity_score', ascending=False)
    
    # --- NOUVEAU 1 : Menu déroulant (Filtre par marque) ---
    marques_disponibles = sorted(df_nat['brands'].dropna().unique())
    marques_selectionnees = st.multiselect(
        "🔍 Filtrer par marque(s) :",
        options=marques_disponibles,
        default=[],
        help="Laissez vide pour voir toutes les marques"
    )
    
    # Application du filtre si des marques sont sélectionnées
    if marques_selectionnees:
        df_nat = df_nat[df_nat['brands'].isin(marques_selectionnees)]
        
    top_100_nat = df_nat.head(100)
    
    # Affichage du tableau
    st.dataframe(top_100_nat[['product_name', 'brands', 'popularity_score']].reset_index(drop=True))

    # --- NOUVEAU 2 : Bouton de téléchargement CSV ---
    csv_data = top_100_nat.to_csv(index=False).encode('utf-8')
    st.download_button(
        label="📥 Télécharger ce tableau (CSV)",
        data=csv_data,
        file_name='produits_naturels_filtres.csv',
        mime='text/csv',
    )


# --- SECTION 3 : COMPOSANTS ET ORIGINES (Top 100 Naturels) ---
st.header("3. Composition et Approvisionnement")

# Extraction et nettoyage des ingrédients
all_ingredients = []
for tags in top_100_nat['ingredients_tags']:
    if isinstance(tags, (list, np.ndarray)):
        for tag in tags:
            tag_str = str(tag)
            clean_tag = tag_str.split(':')[-1]
            clean_tag = clean_tag.replace('-', ' ').strip().lower()
            if clean_tag:
                all_ingredients.append(clean_tag.capitalize())

# Extraction et nettoyage des origines
all_origins = []
traductions_pays = {
    'france': 'France', 'morocco': 'Maroc', 'spain': 'Espagne',
    'switzerland': 'Suisse', 'belgium': 'Belgique', 'denmark': 'Danemark',
    'italy': 'Italie', 'italie': 'Italie', 'amerique latine': 'Amérique Latine',
    'germany': 'Allemagne', 'uk': 'Royaume-Uni', 'usa': 'États-Unis',
    'united states': 'États-Unis'
}
mots_a_exclure = ['extra', 'virgin', 'oil', 'organic', 'bio', 'agriculture', 'fair trade']

for tags in top_100_nat['origins_tags']:
    if isinstance(tags, (list, np.ndarray)):
        for tag in tags:
            tag_str = str(tag)
            clean_tag = tag_str.split(':')[-1]
            clean_tag = clean_tag.replace('-', ' ').strip().lower()
            if clean_tag and not any(mot in clean_tag for mot in mots_a_exclure):
                final_tag = traductions_pays.get(clean_tag, clean_tag.title())
                all_origins.append(final_tag)

col_c, col_d = st.columns(2)

with col_c:
    st.subheader("Top 20 des Composants (Ingrédients)")
    if all_ingredients:
        top_ingredients = pd.DataFrame.from_dict(Counter(all_ingredients), orient='index', columns=['Occurrences']).sort_values(by='Occurrences', ascending=False).head(20)
        st.bar_chart(top_ingredients)
    else:
        st.info("Pas assez de données d'ingrédients pour ces produits.")

with col_d:
    st.subheader("Espaces Géographiques d'Approvisionnement")
    if all_origins:
        top_origins = pd.DataFrame.from_dict(Counter(all_origins), orient='index', columns=['Occurrences']).sort_values(by='Occurrences', ascending=False).head(10)
        # On réinitialise l'index pour que Altair puisse lire les colonnes correctement
        top_origins = top_origins.reset_index().rename(columns={'index': 'Pays'})
        
        # --- NOUVEAU 3 : Graphique en Camembert (Pie Chart) avec Altair ---
        pie_chart = alt.Chart(top_origins).mark_arc(innerRadius=0).encode(
            theta=alt.Theta(field="Occurrences", type="quantitative"),
            color=alt.Color(field="Pays", type="nominal", legend=alt.Legend(title="Pays d'origine")),
            tooltip=['Pays', 'Occurrences'] # Ajoute une info-bulle au survol de la souris
        ).properties(
            height=350 # Hauteur ajustée pour bien s'aligner avec le graphique des ingrédients
        )
        
        st.altair_chart(pie_chart, use_container_width=True)
    else:
        st.info("La communauté n'a pas renseigné l'origine géographique pour ces produits spécifiques.")

# --- SECTION 4 : ANALYSES AVANCÉES DU MARCHÉ ---
st.header("4. Tendances du Marché & Segments")

col_e, col_f = st.columns(2)

with col_e:
    st.subheader("Les Segments les plus représentés")
    
    all_categories = []
    
    # 1. Les tags inutiles à bannir totalement
    categories_a_ignorer = [
        'open beauty facts', 'non food products', 'hygiene', 'hygiène',
        'cosmetics', 'cosmétiques', 'produits de beauté', 'cosmetic products'
    ]
    
    # 2. Un dictionnaire pour traduire et fusionner les vrais segments
    traduction_categories = {
        'hair': 'Soins Capillaires',
        'shampoos': 'Shampooings',
        'shampooings': 'Shampooings',
        'showers and baths': 'Gels Douche & Bains',
        'shower gels': 'Gels Douche & Bains',
        'face': 'Soins Visage',
        'deodorants': 'Déodorants',
        'déodorants': 'Déodorants',
        'body': 'Soins Corps',
        'soaps': 'Savons',
        'toothpastes': 'Dentifrices',
        'suncare': 'Protections Solaires',
        'in sun protections': 'Protections Solaires'
    }

    # Extraction et nettoyage
    for tags in df['categories_tags']:
        if isinstance(tags, (list, np.ndarray)):
            for tag in tags:
                # Nettoyage de base
                clean_tag = str(tag).split(':')[-1].replace('-', ' ').strip().lower()
                
                # Si le tag n'est pas vide et n'est pas dans notre liste noire
                if clean_tag and clean_tag not in categories_a_ignorer:
                    # On le traduit s'il est dans notre dico, sinon on le met juste en majuscule
                    final_tag = traduction_categories.get(clean_tag, clean_tag.capitalize())
                    all_categories.append(final_tag)
                    
    if all_categories:
        # On affiche le Top 10 des vrais segments
        top_categories = pd.DataFrame.from_dict(Counter(all_categories), orient='index', columns=['Nombre de produits']).sort_values(by='Nombre de produits', ascending=False).head(10)
        st.bar_chart(top_categories)

with col_f:
    st.subheader("Comparaison : Nombre moyen d'ingrédients")
    
    # Créer une colonne avec le nombre d'ingrédients pour chaque produit
    df['nb_ingredients'] = df['ingredients_tags'].apply(lambda x: len(x) if isinstance(x, (list, np.ndarray)) else 0)
    
    # Filtrer pour ne garder que les produits où on a vraiment des ingrédients renseignés (nb > 0)
    df_valid_ing = df[df['nb_ingredients'] > 0]
    
    # Calcul des moyennes
    moyenne_naturels = df_valid_ing[df_valid_ing['is_natural'] == True]['nb_ingredients'].mean()
    moyenne_classiques = df_valid_ing[df_valid_ing['is_natural'] == False]['nb_ingredients'].mean()
    
    # Affichage des KPIs
    st.write("La tendance du marché est-elle au minimalisme ?")
    st.metric(label="Produits Conventionnels", value=f"{moyenne_classiques:.1f} ingrédients")
    st.metric(label="Produits Naturels/Bio", value=f"{moyenne_naturels:.1f} ingrédients", delta=f"{moyenne_naturels - moyenne_classiques:.1f} (formules plus courtes)", delta_color="normal")
    
    # Petit mot d'analyse automatique
    if moyenne_naturels < moyenne_classiques:
        st.success("💡 Analyse : Les produits naturels ont effectivement des formules plus courtes et épurées que les produits conventionnels.")
    else:
        st.info("💡 Analyse : Les produits naturels ont des formulations aussi complexes que les produits conventionnels.")