import os
import pandas as pd
import random

# Parameter
input_root = "../datasets"                  # Eingabe-Ordner mit CSV-Dateien
output_root = "../randomSubsets"              # Zielordner für Subsets
n_subsets = 1                         # Anzahl zufälliger Subsets pro Datei
subset_size = 10000                   # Zeilen pro Subset

# Gehe rekursiv durch alle Ordner und suche CSV-Dateien
for dataset_name in os.listdir(input_root):
    dataset_path = os.path.join(input_root, dataset_name)
    if not os.path.isdir(dataset_path):
        continue

    for filename in os.listdir(dataset_path):
        if not filename.lower().endswith(".csv"):
            continue

        file_path = os.path.join(dataset_path, filename)
        print(f"Verarbeite: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Fehler beim Laden: {e}")
            continue

        if len(df) < subset_size:
            print(f"Datei hat nur {len(df)} Zeilen – zu wenig, wird übersprungen.")
            continue

        # Zielordner: ./subsets/<dataset_name>/<filename_ohne_csv>/
        base_name = os.path.splitext(filename)[0]
        output_dir = os.path.join(output_root, dataset_name, base_name)
        os.makedirs(output_dir, exist_ok=True)

        # Erstelle Subsets
        for i in range(n_subsets):
            subset = df.sample(n=subset_size, random_state=i)
            out_path = os.path.join(output_dir, f"subset_{i+1}.csv")
            subset.to_csv(out_path, index=False)
            print(f"subset_{i+1}.csv gespeichert unter {out_path}")
