import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse
import matplotlib as mpl

def visualize_metric_2d_improved():
    """
    Génère un graphique 2D visualisant la métrique anisotrope de manière plus lisible.
    """
    # Paramètres de la métrique basés sur le code Rust
    STRETCH_MAGNITUDE = 10.0
    PERP_MAGNITUDE = 1.0
    
    # Création d'un maillage de points pour la visualisation
    x = np.linspace(0.0, 2.0, 15)
    y = np.linspace(0.0, 1.0, 8)
    xx, yy = np.meshgrid(x, y)

    # Création de la figure
    fig, ax = plt.subplots(figsize=(12, 8))

    # Facteur d'échelle pour les ellipses afin d'éviter le chevauchement
    scale_factor = 0.05
    
    for i in range(xx.shape[0]):
        for j in range(xx.shape[1]):
            p = np.array([xx[i, j], yy[i, j]])
            
            # Métrique par défaut (isotrope)
            metric_matrix = np.identity(2)
            
            # Appliquer la logique de la métrique anisotrope du code Rust
            if np.abs(p[0] - 1.0) < 0.1 and np.abs(p[1] - 0.5) < 0.3:
                dist_to_center_y = np.abs(p[1] - 0.5)
                influence = 1.0 - (dist_to_center_y / 0.3)
                
                # Définir une matrice de métrique anisotrope étirée selon l'axe X
                # (correspondant à l'axe Z dans le code 3D)
                # Notez que le STRETCH_MAGNITUDE est sur la diagonale (y) et le PERP_MAGNITUDE sur le (x)
                # pour que l'ellipse s'étire horizontalement dans la visualisation.
                metric_matrix = np.array([
                    [PERP_MAGNITUDE, 0],
                    [0, STRETCH_MAGNITUDE * influence]
                ])
                
            # Inversion de la métrique pour le calcul des axes de l'ellipse
            H = np.linalg.inv(metric_matrix)
            eigvals, eigvecs = np.linalg.eigh(H)
            
            # Calcul de la largeur, hauteur et de l'angle de l'ellipse
            width = 2.0 * np.sqrt(eigvals[0]) * scale_factor
            height = 2.0 * np.sqrt(eigvals[1]) * scale_factor
            
            angle = np.degrees(np.arctan2(eigvecs[1, 0], eigvecs[0, 0]))
            
            color = 'red' if np.abs(p[0] - 1.0) < 0.1 and np.abs(p[1] - 0.5) < 0.3 else 'blue'

            # Création et ajout de l'ellipse au graphique
            ellipse = Ellipse(
                (p[0], p[1]),
                width=width,
                height=height,
                angle=angle,
                color=color,
                alpha=0.6,
                zorder=2
            )
            ax.add_patch(ellipse)
    
    # Ajustement des limites pour une meilleure visualisation
    ax.set_title("Visualisation d'une métrique Anisotrope en 2D")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlim(x.min() - 0.1, x.max() + 0.1)
    ax.set_ylim(y.min() - 0.1, y.max() + 0.1)
    ax.grid(True)
    
    # Légende pour les couleurs
    iso_patch = mpl.patches.Patch(color='blue', alpha=0.6, label='Métrique isotrope')
    aniso_patch = mpl.patches.Patch(color='red', alpha=1.0, label='Métrique anisotrope')
    ax.legend(handles=[iso_patch, aniso_patch])

    # Sauvegarde du graphique
    plt.savefig("Metric_plot_2d_improved_corrected.png")
    plt.show()

# Exécution du script
visualize_metric_2d_improved()