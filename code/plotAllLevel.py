import pandas as pd
import matplotlib.pyplot as plt
import re

# CSV-Datei laden
df = pd.read_csv("../results/kCenter/AusLogkCenter1.csv")

# Zeilen mit fehlender Accuracy ignorieren
df = df.dropna(subset=["test_acc"])

# Funktion zum Extrahieren von Typ und Level aus subset_path
def extract_type_level(path):
    path = path.replace(" ", "_").replace(".", "")
    match = re.search(r"_(Outlier|GaussianFeatureNoise|MissingCompletely|ClassImbalancedness)_(std|ratio)(\d+)", path)
    if match:
        return match.group(1), int(match.group(3))
    return None, None

# Neue Spalten mit Typ und Level erzeugen
df[['type', 'level']] = df['subset_path'].apply(lambda p: pd.Series(extract_type_level(p)))

# Relevante Scores
score_cols = ['test_acc', 'balanced_acc', 'precision', 'recall', 'f1', 'roc_auc', 'mcc', 'kappa']

# Gruppiert nach Typ und Level den Mittelwert je Score berechnen
grouped = df.groupby(['type', 'level'])[score_cols].mean().reset_index()

# Plot für Outlier
subset = grouped[grouped['type'] == 'Outlier']
plt.figure(figsize=(10, 6))
for col in score_cols:
    plt.plot(subset['level'], subset[col], marker='o', label=col)
plt.title("Outlier: Scores vs. Level")
plt.xlabel("Noise Level (std %)")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot für GaussianFeatureNoise
subset = grouped[grouped['type'] == 'GaussianFeatureNoise']
plt.figure(figsize=(10, 6))
for col in score_cols:
    plt.plot(subset['level'], subset[col], marker='o', label=col)
plt.title("Gaussian Feature Noise: Scores vs. Level")
plt.xlabel("Noise Level (std %)")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot für MissingCompletely
subset = grouped[grouped['type'] == 'MissingCompletely']
plt.figure(figsize=(10, 6))
for col in score_cols:
    plt.plot(subset['level'], subset[col], marker='o', label=col)
plt.title("Missing Completely: Scores vs. Level")
plt.xlabel("Missing Ratio (%)")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()

# Plot für ClassImbalancedness
subset = grouped[grouped['type'] == 'ClassImbalancedness']
plt.figure(figsize=(10, 6))
for col in score_cols:
    plt.plot(subset['level'], subset[col], marker='o', label=col)
plt.title("Class Imbalancedness: Scores vs. Level")
plt.xlabel("Class Ratio")
plt.ylabel("Score")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
