import pandas as pd
import matplotlib.pyplot as plt
import os
import re
import numpy as np

# --- Configuration (doit correspondre à vos répertoires de logs) ---
LOG_ROOT_DIR = "remesh_logs_full_config"
LOG_DIR_2D = os.path.join(LOG_ROOT_DIR, "remesh_stats_2D")
LOG_DIR_3D = os.path.join(LOG_ROOT_DIR, "remesh_stats_3D")
OUTPUT_PLOTS_DIR = "remesh_plots"
os.makedirs(OUTPUT_PLOTS_DIR, exist_ok=True)


# --- Fonction de parsing de temps plus robuste ---
def parse_time_string_robust(time_str):
    time_str = str(time_str).strip().lower()
    if 'e' in time_str:
        try:
            return float(time_str)
        except (ValueError, TypeError):
            return np.nan
    elif time_str.endswith('ms'):
        try:
            return float(time_str[:-2]) / 1000.0
        except (ValueError, TypeError):
            return np.nan
    elif time_str.endswith('s'):
        try:
            return float(time_str[:-1])
        except (ValueError, TypeError):
            return np.nan
    else:
        try:
            return float(time_str)
        except (ValueError, TypeError):
            return np.nan

# --- Fonction pour lire et parser les logs ---
def parse_log_file(filepath):
    """
    Analyse un fichier de log et extrait les lignes de données formatées,
    en gérant les lignes de données qui sont divisées sur plusieurs lignes.
    """
    data_records = []
    current_data_line = ""
    partition_times = []
    with open(filepath, 'r') as f:
        for line in f:
            stripped_line = line.strip()
            
            if stripped_line.startswith("partition "):
                try:
                    parts = stripped_line.split(',')
                    time_str = parts[1].strip()
                    time_good = time_str.split(' ')
                    time_val = parse_time_string_robust(time_good[0])
                    partition_times.append(time_val)
                except (IndexError, ValueError):
                    continue

            # Si la ligne commence par "DATA,", c'est le début d'un nouveau bloc de données.
            if stripped_line.startswith("DATA,"):
                if current_data_line:
                    parts = current_data_line.replace('DATA,', '', 1).split(',')
                    record = {}
                    for part in parts:
                        if '=' in part:
                            key, value = part.split('=', 1)
                            record[key.strip()] = value.strip()
                    
                    record['partition_times'] = partition_times
                    data_records.append(record)
                    
                current_data_line = stripped_line
            elif current_data_line and stripped_line:
                current_data_line += stripped_line.replace(" ", "")
        
        if current_data_line:
            parts = current_data_line.replace('DATA,', '', 1).split(',')
            record = {}
            for part in parts:
                if '=' in part:
                    key, value = part.split('=', 1)
                    record[key.strip()] = value.strip()
            
            record['partition_times'] = partition_times
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
numeric_cols = ['D', 'num_elements', 'remeshing_partition_time', 'remeshing_ifc_time', 'remeshing_ptime_imbalance', 'total_elapsed_time']
for col in numeric_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.replace('?', '', regex=False).apply(parse_time_string_robust)
    else:
        print(f"Avertissement : la colonne '{col}' n'est pas présente dans le DataFrame. Elle pourrait ne pas apparaître dans les logs.")

# NOUVEAU : Conversion de la colonne 'option' en booléen
if 'option' in df.columns:
    df['option'] = df['option'].astype(str).str.lower().str.strip() == 'true'
    # Ajout de la colonne 'metric_op'
    df['metric_op'] = df['option'].apply(lambda x: 'collapse' if x else 'split')
else:
    print("Avertissement : la colonne 'option' n'est pas présente. La distinction collapse/split ne sera pas faite.")
    df['metric_op'] = 'split'

# --- NOUVEAU : Calculer le temps total de remaillage et le temps fixe ---
df['total_remeshing_time'] = df['remeshing_partition_time'] + df['remeshing_ifc_time']
df.dropna(subset=['D', 'num_elements', 'total_remeshing_time'], inplace=True)

# print("\n--- Aperçu des données collectées ---")
# print(df.head())
# print("\n--- Informations sur les données ---")
# print(df.info())

# --- Agrégation des données ---
grouped_stats = df.groupby(['D', 'metric_type', 'cost_estimator', 'partitioner', 'num_elements', 'metric_op']).agg(
    total_remeshing_time_mean=('total_remeshing_time', 'mean'),
    total_remeshing_time_std=('total_remeshing_time', 'std'),
    remeshing_partition_time_mean=('remeshing_partition_time', 'mean'),
    remeshing_partition_time_std=('remeshing_partition_time', 'std'),
    remeshing_ifc_time_mean=('remeshing_ifc_time', 'mean'),
    remeshing_ifc_time_std=('remeshing_ifc_time', 'std')
).reset_index()

# Génération des graphiques de temps total (comme dans votre version originale)
print("\n--- Génération des graphiques de comparaison Nocost vs Toto avec erreurs ---")
plot_grouped = grouped_stats.groupby(['D', 'metric_type', 'partitioner', 'metric_op'])

for name, group in plot_grouped:
    dimension, metric_type, partitioner, metric_op = name
    
    nocost_data = group[group['cost_estimator'] == 'Nocost'].sort_values('num_elements')
    totocost_data = group[group['cost_estimator'] == 'Toto'].sort_values('num_elements')

    if nocost_data.empty and totocost_data.empty:
        continue

    plt.figure(figsize=(12, 7))
    if not nocost_data.empty:
        plt.plot(nocost_data['num_elements'], nocost_data['remeshing_partition_time_mean'],
                 '--', color='orange', label='Temps partition (NoCost)', alpha=0.6)
        plt.fill_between(nocost_data['num_elements'], 
                         nocost_data['remeshing_partition_time_mean'] - nocost_data['remeshing_partition_time_std'],
                         nocost_data['remeshing_partition_time_mean'] + nocost_data['remeshing_partition_time_std'],
                         color='orange', alpha=0.2)
        plt.plot(nocost_data['num_elements'], nocost_data['remeshing_ifc_time_mean'],
                 '--', color='red', label='Temps interface (NoCost)', alpha=0.6)
        plt.fill_between(nocost_data['num_elements'], 
                         nocost_data['remeshing_ifc_time_mean'] - nocost_data['remeshing_ifc_time_std'],
                         nocost_data['remeshing_ifc_time_mean'] + nocost_data['remeshing_ifc_time_std'],
                         color='red', alpha=0.2)
    if not totocost_data.empty:
        plt.plot(totocost_data['num_elements'], totocost_data['remeshing_partition_time_mean'],
                 '--', color='cyan', label='Temps partition (TotoCost)', alpha=0.6)
        plt.fill_between(totocost_data['num_elements'], 
                         totocost_data['remeshing_partition_time_mean'] - totocost_data['remeshing_partition_time_std'],
                         totocost_data['remeshing_partition_time_mean'] + totocost_data['remeshing_partition_time_std'],
                         color='cyan', alpha=0.2)
        plt.plot(totocost_data['num_elements'], totocost_data['remeshing_ifc_time_mean'],
                 '--', color='blue', label='Temps interface (TotoCost)', alpha=0.6)
        plt.fill_between(totocost_data['num_elements'], 
                         totocost_data['remeshing_ifc_time_mean'] - totocost_data['remeshing_ifc_time_std'],
                         totocost_data['remeshing_ifc_time_mean'] + totocost_data['remeshing_ifc_time_std'],
                         color='blue', alpha=0.2)

    if not nocost_data.empty:
        plt.errorbar(nocost_data['num_elements'], nocost_data['total_remeshing_time_mean'],
                     yerr=nocost_data['total_remeshing_time_std'], fmt='-o', capsize=5,
                     label='Total Remeshing Time (NoCost)')
    if not totocost_data.empty:
        plt.errorbar(totocost_data['num_elements'], totocost_data['total_remeshing_time_mean'],
                     yerr=totocost_data['total_remeshing_time_std'], fmt='-x', capsize=5,
                     label='Total Remeshing Time (TotoCost)')

    plt.title(f'Temps de remaillage total vs. Nombre d\'éléments ({dimension}D, {metric_type}, Part: {partitioner}, Op: {metric_op})')
    plt.xlabel('Nombre d\'éléments')
    plt.ylabel('Temps (secondes)')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()

    filename = f"remeshing_time_{dimension}D_{metric_type}_{partitioner}_{metric_op}_comparison_mean_std.png"
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = filename.replace(' ', '_')
    plot_path = os.path.join(OUTPUT_PLOTS_DIR, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Graphique enregistré : {plot_path}")


# --- Génération de graphiques de disparité par partition ---
print("\n--- Génération des graphiques de disparité par partition ---")
plot_grouped_disparity = df.groupby(['D', 'metric_type', 'partitioner', 'metric_op'])

for name, group in plot_grouped_disparity:
    dimension, metric_type, partitioner, metric_op = name
    
    unique_num_elements = sorted(group['num_elements'].unique())
    
    plt.figure(figsize=(12, 7))
    
    x_positions = []
    labels = []
    
    for i, num_elements in enumerate(unique_num_elements):
        
        # Données de NoCost pour le nombre d'éléments actuel
        nocost_data = group[(group['num_elements'] == num_elements) & (group['cost_estimator'] == 'Nocost')]
        
        # Données de TotoCost pour le nombre d'éléments actuel
        totocost_data = group[(group['num_elements'] == num_elements) & (group['cost_estimator'] == 'Toto')]
        
        # Tracé des points pour NoCost à une position 2*i
        if not nocost_data.empty:
            for _, row in nocost_data.iterrows():
                partition_times = row['partition_times']
                x_values = np.full(len(partition_times), 2*i)
                plt.scatter(x_values, partition_times, c='orange', marker='o', s=50, alpha=0.6)
            x_positions.append(2*i)
            labels.append(f'NoCost\n({num_elements})')
        
        # Tracé des points pour TotoCost à une position 2*i + 1
        if not totocost_data.empty:
            for _, row in totocost_data.iterrows():
                partition_times = row['partition_times']
                x_values = np.full(len(partition_times), 2*i + 1)
                plt.scatter(x_values, partition_times, c='cyan', marker='x', s=50, alpha=0.6)
            # Ajoute la position et l'étiquette seulement si les données existent
            x_positions.append(2*i + 1)
            labels.append(f'TotoCost\n({num_elements})')

    if not x_positions:
        plt.close()
        continue
        
    plt.xticks(x_positions, labels)
    
    # Créer les légendes à la main car scatter ne les gère pas de manière simple dans cette boucle
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='orange', markersize=10, label='NoCost'),
        plt.Line2D([0], [0], marker='x', color='w', markerfacecolor='cyan', markeredgecolor='cyan', markersize=10, label='TotoCost')
    ]
    
    plt.title(f'Disparité des temps de remaillage par partition ({dimension}D, {metric_type}, Part: {partitioner}, Op: {metric_op})')
    plt.xlabel('Estimateur de coût et Nombre d\'éléments')
    plt.ylabel('Temps de remaillage par partition (secondes)')
    plt.grid(True)
    plt.legend(handles=legend_elements)
    plt.tight_layout()

    filename = f"partition_disparity_scatter_{dimension}D_{metric_type}_{partitioner}_{metric_op}_comparison.png"
    filename = re.sub(r'[^\w\s.-]', '', filename)
    filename = filename.replace(' ', '_')
    plot_path = os.path.join(OUTPUT_PLOTS_DIR, filename)
    plt.savefig(plot_path)
    plt.close()
    print(f"Graphique de disparité enregistré : {plot_path}")

print("\nAnalyse terminée. Les graphiques sont disponibles dans le répertoire 'remesh_plots'.")