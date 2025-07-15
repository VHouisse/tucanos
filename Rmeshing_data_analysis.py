import pandas as pd
import matplotlib.pyplot as plt
import os
import re

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
                # Supprime le préfixe "DATA," et divise la ligne par virgule
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

# --- Nettoyage et conversion des types ---
numeric_cols = ['D', 'num_splits', 'num_elements', 'time_seconds']
for col in numeric_cols:
    if col == 'time_seconds':
        # Gère les valeurs comme "71.62365ms" en retirant 'ms' et convertissant en float (puis en secondes)
        df[col] = df[col].astype(str).str.replace('ms', '', regex=False)
        df[col] = pd.to_numeric(df[col], errors='coerce') / 1000.0 # Convertit les ms en secondes
    else:
        df[col] = pd.to_numeric(df[col], errors='coerce')

# Supprime les lignes où les conversions numériques ont échoué
df.dropna(subset=numeric_cols, inplace=True)

# Nettoie les noms de types d'éléments Rust (ex: "tucanos::mesh::Triangle" -> "Triangle")
df['E'] = df['E'].apply(lambda x: x.split('::')[-1] if isinstance(x, str) else x)

print("\n--- Aperçu des données collectées ---")
print(df.head())
print("\n--- Informations sur les données ---")
print(df.info())

# --- Génération de graphiques ---

print("\n--- Génération des graphiques de comparaison Nocost vs Toto ---")

# Regroupe les données par dimension, type de métrique et partitionneur
grouped = df.groupby(['D', 'metric_type', 'partitioner'])

for name, group in grouped:
    dimension, metric_type, partitioner = name
    
    # Filtre les données pour les estimateurs de coût 'Nocost' et 'Toto'
    nocost_data = group[group['cost_estimator'] == 'Nocost'].sort_values('num_elements')
    toto_data = group[group['cost_estimator'] == 'Toto'].sort_values('num_elements')

    if nocost_data.empty and toto_data.empty:
        continue # Passe si aucune donnée pour cette combinaison

    plt.figure(figsize=(10, 6))

    if not nocost_data.empty:
        plt.plot(nocost_data['num_elements'], nocost_data['time_seconds'], 
                 marker='o', label='NoCost Estimator')
    
    if not toto_data.empty:
        plt.plot(toto_data['num_elements'], toto_data['time_seconds'], 
                 marker='x', label='TotoCost Estimator')
    
    plt.title(f'Temps d\'exécution vs. Nombre d\'éléments ({dimension}D, {metric_type}, Part: {partitioner})')
    plt.xlabel('Nombre d\'éléments')
    plt.ylabel('Temps d\'exécution (secondes)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    # Nettoie le nom de fichier pour éviter les caractères spéciaux
    filename = f"exec_time_{dimension}D_{metric_type}_{partitioner}_comparison.png"
    filename = re.sub(r'[^\w\s.-]', '', filename) # Supprime les caractères non alphanumériques (sauf alphanumériques, espaces, tirets, points)
    filename = filename.replace(' ', '_') # Remplace les espaces par des underscores
    
    plot_path = os.path.join(OUTPUT_PLOTS_DIR, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Graphique enregistré : {plot_path}")

print("\nAnalyse terminée. Les graphiques sont disponibles dans le répertoire 'remesh_plots'.")