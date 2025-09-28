import pandas as pd
import numpy as np
from sklearn.metrics import pairwise_distances_argmin_min
from sklearn.preprocessing import StandardScaler
from pathlib import Path
import os

def k_center_greedy(X, k):
    print("kcenter")
    """Implementiert den k-Center Greedy Algorithmus für Subset Selection"""
    idx = [np.random.randint(len(X))]
    while len(idx) < k:
        _, distances = pairwise_distances_argmin_min(X, X[idx])
        idx.append(np.argmax(distances))
        print(size(idx))
    return np.array(idx)

def create_kcenter_subsets(dataset_name, base_path, k=500):
    """
    Erstellt k-Center Subsets für einen gegebenen Datensatz
    
    Args:
        dataset_name (str): Name des Datensatzes
        base_path (Path): Basis-Pfad zum Projekt
        k (int): Größe des Subsets
    """
    input_path = base_path / "datasets" / dataset_name
    output_path = base_path / "kCenterSubsets" / dataset_name
    
    # Erstelle Output-Verzeichnis
    output_path.mkdir(parents=True, exist_ok=True)

    # Iteriere über alle CSV-Dateien im Dataset-Ordner
    for filename in os.listdir(input_path):
        if not filename.endswith(".csv"):
            print('no file under this filename')
            continue

        file_path = input_path / filename
        out_file = output_path / filename.replace(".csv", f"_kcenter{k}.csv")

        # continie if file already exists
        if out_file.exists():
            print('out_file')
            continue

        try:
            print('try')
            df = pd.read_csv(file_path)

            # Features und Label trennen (Label = letzte Spalte)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            # Normalisieren
            X_scaled = StandardScaler().fit_transform(X)

            # Auswahl mit k-Center
            current_k = min(k, len(X_scaled))
            selected_idx = k_center_greedy(X_scaled, k=current_k)
            df_subset = df.iloc[selected_idx]

            # Speichern
            df_subset.to_csv(out_file, index=False)
            print(f"✅ {filename} → Subset gespeichert ({len(selected_idx)} Punkte)")
        except Exception as e:
            print(f"⚠️ Fehler bei {file_path}: {e}")