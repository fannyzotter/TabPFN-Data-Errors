import pandas as pd
import matplotlib.pyplot as plt
import re

# CSV-Datei laden
df = pd.read_csv("../results/AusLog10Samples.csv")
#df2 = pd.read_csv("../results/Aus300Log.csv")
#set column names as there are no headers in the csv
df.columns = ["dataset", "subset_path", "subset_name", "n_samples", "n_features", "test_acc", "inference_time_sec", "status"]
#df2
# Zeilen zusammenführen
#df = pd.concat([df1, df2], ignore_index=True)
# Zeilen mit fehlender Accuracy ignorieren
df = df.dropna(subset=["test_acc"])

# Funktion zum Extrahieren von Typ und Level aus subset_path
def extract_type_level(path):
    match = re.search(r"_(Outlier|GaussianFeatureNoise|MissingCompletely|ClassImbalancedness)_(std|ratio)(\d+)", path)
    if match:
        return match.group(1), int(match.group(3))
    return None, None

# Neue Spalten mit Typ und Level erzeugen
df[['type', 'level']] = df['subset_path'].apply(lambda p: pd.Series(extract_type_level(p)))

# Durchschnittliche Accuracy pro Typ und Level berechnen
grouped = df.groupby(['type', 'level'])['test_acc'].mean().reset_index()

# Plot
plt.figure(figsize=(8, 5))

for perturbation_type in ['Outlier', 'GaussianFeatureNoise', 'MissingCompletely', 'ClassImbalancedness']:
    subset = grouped[grouped['type'] == perturbation_type]
    plt.plot(subset['level'], subset['test_acc'], marker='o', label=perturbation_type)

plt.xlabel("Noise Level (std %)")
plt.ylabel("Durchschnittliche Test Accuracy")
plt.title("Durchschnittliche Accuracy vs. Störungs-Level")
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
