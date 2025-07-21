import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

path = "../randomSubsets/Australian/Australian_ClassImbalancedness_ratio0_54/subset_1.csv"
# Daten laden
df = pd.read_csv(path)  # Pfad ggf. anpassen

# Setup Grid
fig, axes = plt.subplots(3, 5, figsize=(20, 12))
axes = axes.flatten()

# Plot für aufeinanderfolgende Feature-Paare (A1 vs A2, ..., A13 vs A14)
for i in range(13):  # 13 Paare
    x_feat = f"A{i+1}"
    y_feat = f"A{i+2}"
    ax = axes[i]

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
fig.delaxes(axes[13])
fig.delaxes(axes[14])

# Gemeinsame Legende
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Klasse (A15)", loc='upper right', bbox_to_anchor=(1.12, 0.95))

plt.tight_layout()
name = path.split("/")[-1]
plt.suptitle(f"Scatter Plots of Features in {name}", fontsize=16)
plt.title(f"Plot file: {name}", fontsize=10, loc='left')
plt.show()

