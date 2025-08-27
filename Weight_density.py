import re
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

def analyze_log_weights(file_name):
    """
    Analyse un fichier de log pour extraire les poids,
    calculer les statistiques et créer un graphique de densité.
    """
    all_weights = []

    try:
        with open(file_name, 'r') as file:
            for line in file:
                # Cherche les lignes qui commencent par 'Weights ['
                if line.strip().startswith('Weights ['):
                    # Utilise une expression régulière pour extraire la chaîne de nombres
                    match = re.search(r'Weights \[(.*?)\]', line)
                    if match:
                        # Divise la chaîne en nombres et les convertit en float
                        weights_str = match.group(1)
                        weights = [float(w.strip()) for w in weights_str.split(',')]
                        all_weights.extend(weights)
    except FileNotFoundError:
        print(f"Erreur : le fichier '{file_name}' n'a pas été trouvé.")
        return

    if not all_weights:
        print("Aucune donnée de poids trouvée. Assurez-vous que le fichier contient des lignes qui commencent par 'Weights [...'.")
        return

    # Convertit la liste en tableau numpy pour les calculs
    weights_np = np.array(all_weights)

    # Calcule les statistiques demandées
    total_sum = np.sum(weights_np)
    median_p50 = np.percentile(weights_np, 50)
    median_p75 = np.percentile(weights_np, 75)

    print(f"Somme totale de tous les éléments : {total_sum}")
    print(f"Médiane (50e centile) : {median_p50}")
    print(f"75e centile : {median_p75}")

    # Crée et sauvegarde le graphique de densité
    plt.figure(figsize=(10, 6))
    sns.kdeplot(weights_np, fill=True)
    plt.title('Distribution de la densité des poids', fontsize=16)
    plt.xlabel('Valeurs des poids', fontsize=12)
    plt.ylabel('Densité', fontsize=12)
    plt.grid(True)
    plt.savefig('density_plot.png')
    plt.close()

    # Sauvegarde les données dans un fichier CSV pour plus de commodité
    df_weights = pd.DataFrame(all_weights, columns=['Weights'])
    df_weights.to_csv('weights_data.csv', index=False)
    print("Graphique de densité et fichier CSV des données créés avec succès.")

# Exécute la fonction avec le nom de votre fichier
analyze_log_weights('log_weights.txt')