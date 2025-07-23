import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np # Importer numpy pour les calculs statistiques

# --- Configuration (doit correspondre à vos répertoires de logs) ---
LOG_ROOT_DIR = "remesh_logs_full_config"
LOG_DIR_2D = os.path.join(LOG_ROOT_DIR, "remesh_stats_2D")
LOG_DIR_3D = os.path.join(LOG_ROOT_DIR, "remesh_stats_3D")
OUTPUT_PLOTS_DIR = "remesh_plots"
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)

# --- Fonction pour lire et parser les logs ---
def parse_log_file(filepath):
    """
    Analyse un fichier de log et extrait les lignes de données formatées.
    Adapte la lecture au format "DATA,key=value,key=value,..."
    """
    data_records = []
    with open(filepath, 'r') as f:
        for line in f:
            if line.startswith("DATA,"):
                parts = line[5:].strip().split(',')
                record = {}
                for part in parts:
                    if '=' in part:
                        key, value = part.split('=', 1)
                        record[key] = value
                data_records.append(record)
    return data_records

# --- Collecter toutes les données ---
all_data = []

print(f"Collecte des données depuis {LOG_DIR_2D}...")
for root, _, files in os.walk(LOG_DIR_2D):
    for file in files:
        if file.endswith(".txt"):
            filepath = os.path.join(root, file)
            all_data.extend(parse_log_file(filepath))

print(f"Collecte des données depuis {LOG_DIR_3D}...")
for root, _, files in os.walk(LOG_DIR_3D):
    for file in files:
        if file.endswith(".txt"):
            filepath = os.path.join(root, file)
            all_data.extend(parse_log_file(filepath))

if not all_data:
    print("Aucune donnée trouvée dans les fichiers de log. Veuillez vous assurer que le programme Rust s'exécute et journalise correctement.")
    exit()

df = pd.DataFrame(all_data)

# --- CORRECTION ICI : Fonction de parsing de temps plus robuste ---
def parse_time_string_robust(time_str):
    time_str = str(time_str).strip().lower() # Convertir en string, enlever espaces, minuscules
    if time_str.endswith('ms'):
        try:
            return float(time_str[:-2]) / 1000.0
        except ValueError:
            return np.nan # Retourne NaN si la conversion échoue
    elif time_str.endswith('s'):
        try:
            return float(time_str[:-1])
        except ValueError:
            return np.nan
    else: # Si aucune unité, tente de parser comme un float direct
        try:
            return float(time_str)
        except ValueError:
            return np.nan # Retourne NaN si ce n'est pas un nombre valide

# --- Nettoyage et conversion des types ---
numeric_cols = ['D','num_elements', 'time_seconds']

# Appliquer la fonction de parsing robuste spécifiquement à la colonne time_seconds
df['time_seconds'] = df['time_seconds'].apply(parse_time_string_robust)

# Convertir les autres colonnes numériques (D, num_elements)
for col in ['D', 'num_elements']:
    df[col] = pd.to_numeric(df[col], errors='coerce')

# Supprime les lignes où les conversions numériques ont échoué
df.dropna(subset=numeric_cols, inplace=True)

print("\n--- Aperçu des données collectées ---")
print(df.head())
print("\n--- Informations sur les données ---")
print(df.info())

# --- NOUVEAU : Agrégation des données ---
# Grouper par toutes les colonnes de configuration sauf l'itération, puis calculer la moyenne et l'écart-type
# On ne se soucie plus de l'index de répétition dans le log, on agrège simplement par les autres paramètres
grouped_stats = df.groupby(['D', 'metric_type', 'cost_estimator', 'partitioner', 'num_elements'])['time_seconds'].agg(['mean', 'std']).reset_index()

# Renommer les colonnes pour plus de clarté
grouped_stats = grouped_stats.rename(columns={'mean': 'time_mean', 'std': 'time_std'})

print("\n--- Aperçu des statistiques agrégées ---")
print(grouped_stats.head())

# --- Génération de graphiques avec moyenne et variance ---
print("\n--- Génération des graphiques de comparaison Nocost vs Toto avec erreurs ---")

# Regrouper pour les graphiques, mais cette fois en utilisant les colonnes de configuration de base
# On itère sur les mêmes groupes que précédemment, mais en utilisant grouped_stats
plot_grouped = grouped_stats.groupby(['D', 'metric_type', 'partitioner'])

for name, group in plot_grouped:
    dimension, metric_type, partitioner = name
    
    # Filtrer les données agrégées pour Nocost et Toto
    nocost_data = group[group['cost_estimator'] == 'Nocost'].sort_values('num_elements')
    toto_data = group[group['cost_estimator'] == 'Toto'].sort_values('num_elements')

    if nocost_data.empty and toto_data.empty:
        continue

    plt.figure(figsize=(12, 7)) # Agrandir légèrement le graphique

    # Tracer Nocost avec barres d'erreur
    if not nocost_data.empty:
        plt.errorbar(nocost_data['num_elements'], nocost_data['time_mean'], 
                     yerr=nocost_data['time_std'], fmt='-o', capsize=5, 
                     label='NoCost Estimator (Moyenne ± Écart-type)')
    
    # Tracer Toto avec barres d'erreur
    if not toto_data.empty:
        plt.errorbar(toto_data['num_elements'], toto_data['time_mean'], 
                     yerr=toto_data['time_std'], fmt='-x', capsize=5, 
                     label='TotoCost Estimator (Moyenne ± Écart-type)')
    
    plt.title(f'Temps d\'exécution moyen vs. Nombre d\'éléments ({dimension}D, {metric_type}, Part: {partitioner})')
    plt.xlabel('Nombre d\'éléments')
    plt.ylabel('Temps d\'exécution (secondes)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = f"exec_time_{dimension}D_{metric_type}_{partitioner}_comparison_mean_std.png"
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = filename.replace(' ', '_')
    
    plot_path = os.path.join(OUTPUT_PLOTS_DIR, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Graphique enregistré : {plot_path}")

print("\nAnalyse terminée. Les graphiques sont disponibles dans le répertoire 'remesh_plots'.")