import os
import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler

# Parameter
input_root = "../datasets"
output_root = "../kCenterSubsets"
k = 500  # Größe des Subsets

def k_center_greedy(X, k):
    idx = [np.random.randint(len(X))]
    while len(idx) < k:
        _, distances = pairwise_distances_argmin_min(X, X[idx])
        idx.append(np.argmax(distances))
    return np.array(idx)

# Erstelle Ausgabeordner falls nicht vorhanden
os.makedirs(output_root, exist_ok=True)

# Iteriere durch alle Datasets
for dataset_name in os.listdir(input_root):
    dataset_path = os.path.join(input_root, dataset_name)
    
    # Nur Ordner berücksichtigen
    if not os.path.isdir(dataset_path):
        continue

    # Erstelle entsprechenden Output-Unterordner
    dataset_out_path = os.path.join(output_root, dataset_name.replace(" ", "_").replace(".", ""))
    os.makedirs(dataset_out_path, exist_ok=True)

    # Iteriere über alle CSV-Dateien im Dataset-Ordner
    for filename in os.listdir(dataset_path):
        if not filename.startswith("Australien") or not filename.endswith(".csv"):
            continue

        file_path = os.path.join(dataset_path, filename)
        out_file = os.path.join(dataset_out_path, filename.replace(".csv", f"_kcenter{k}.csv"))

        try:
            df = pd.read_csv(file_path)

            # Features und Label trennen (Label = letzte Spalte)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            # Normalisieren
            X_scaled = StandardScaler().fit_transform(X)

            # Auswahl mit k-Center
            selected_idx = k_center_greedy(X_scaled, k=min(k, len(X_scaled)))
            df_subset = df.iloc[selected_idx]

            # Speichern
            df_subset.to_csv(out_file, index=False)
            print(f"✅ {filename} → Subset gespeichert ({len(selected_idx)} Punkte)")
        except Exception as e:
            print(f"⚠️ Fehler bei {file_path}: {e}")

