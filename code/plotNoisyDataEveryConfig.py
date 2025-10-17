import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import math

path_dir  = "../randomSubsets/Australian/"

all_configs = ['A2', 'A3', 'A5']

for sub_dir in os.listdir(path_dir):
    path = os.path.join(path_dir, sub_dir)
    for file in os.listdir(path):
        if file.endswith("_1.csv"):
            path = os.path.join(path, file)
            print("Plotting file: ", path)

    # Daten laden
    df = pd.read_csv(path)  # Pfad ggf. anpassen
    
    # Setup Grid

    fig, axes = plt.subplots(1, 3, figsize=(8, 5))
    fig.subplots_adjust(wspace=0.4)  # horizontaler Abstand

    
    # Plot alle existierenden paare (A1 bis A14)
    pairs = list(zip(all_configs, all_configs[1:] + all_configs[:1]))  # rotiert y eins weiter
    
    for a, (x_feat, y_feat) in enumerate(pairs):
        ax = axes[a]
        ax.set_title(f"{x_feat} vs {y_feat}")
        sns.scatterplot(
            data=df,
            x=x_feat,
            y=y_feat,
            hue=df["A15"].astype(str),
            palette="Set1",
            ax=ax,
            legend=False
        )
    
#    plt.tight_layout()
    name = path.split("/")[-2]
    plt.suptitle(f"Scatter Plots of Features in {name}", fontsize=12)
#    plt.title(f"Plot file: {name}", fontsize=10, loc='left')
#    plt.show()
    output_dir = os.path.join(path_dir, "plots")
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(f"{output_dir}/{name}_scatter.png")
    plt.close()

