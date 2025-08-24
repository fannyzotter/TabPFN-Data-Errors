import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = "../randomSubsets/Australian/Australian_GaussianFeatureNoise_std01/subset_1.csv"
# Daten laden
df = pd.read_csv(path)  # Pfad ggf. anpassen

# Setup Grid
fig, axes = plt.subplots(7, 9, figsize=(20, 12))
axes = axes.flatten()

# Plot alle existierenden paare (A1 bis A14)
a = 0
for i in range(1, 11):  # 13 mal
    x_feat = f"A{i+1}"
    for j in range(i+1, 12):
        y_feat = f"A{j+1}"
        ax = axes[a]
        a += 1
        sns.scatterplot(
            data=df,
            x=x_feat,
            y=y_feat,
            hue=df["A15"].astype(str),
            palette="Set1",
            ax=ax,
            legend=False
        )

    ax.set_title(f"{x_feat} vs {y_feat}")

# Letztes Feld (14. Index) löschen
#fig.delaxes(axes[13])
#fig.delaxes(axes[14])

# Gemeinsame Legende
#handles, labels = axes[0].get_legend_handles_labels()
#fig.legend(handles, labels, title="Klasse (A15)", loc='upper right', bbox_to_anchor=(1.12, 0.95))

plt.tight_layout()
name = path.split("/")[-1]
plt.suptitle(f"Scatter Plots of Features in {name}", fontsize=16)
plt.title(f"Plot file: {name}", fontsize=10, loc='left')
plt.show()

