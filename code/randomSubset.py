import os
import pandas as pd
import random

# Parameter
input_root = "../datasets"                  # Eingabe-Ordner mit CSV-Dateien
output_root = "../randomSubsets"              # Zielordner für Subsets
n_subsets = 10                         # Anzahl zufälliger Subsets pro Datei

# Gehe rekursiv durch alle Ordner und suche CSV-Dateien
for dataset_name in os.listdir(input_root):

    dataset_path = os.path.join(input_root, dataset_name)
    dataset_name = dataset_name.replace(" ", "_")
    dataset_name = dataset_name.replace(".", "") # Ersetze Punkte in Datasetnamen

    if not os.path.isdir(dataset_path):
        continue

    for filename in os.listdir(dataset_path):

        subset_size = 500

        # nur Australien datensätze
        if not filename.startswith("Australian") or not filename.endswith(".csv"):
            continue
        print(f"Verarbeite Datei: {filename}")
        if not filename.lower().endswith(".csv"):
            continue
        
        file_path = os.path.join(dataset_path, filename)
        print(f"Verarbeite: {file_path}")

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Fehler beim Laden: {e}")
            continue



        # Zielordner: ./subsets/<dataset_name>/<filename_ohne_csv>/
        base_name = os.path.splitext(filename)[0]
        base_name = base_name.replace(" ", "_")
        base_name = base_name.replace(".", "")  
        output_dir = os.path.join(output_root, dataset_name, base_name)


        if len(df) < subset_size:
            print(f"Datei hat nur {len(df)} Zeilen.")
            output_dir = output_dir + (f"_{len(df)}rows")
            subset_size = len(df)

        os.makedirs(output_dir, exist_ok=True)

        # Erstelle Subsets
        for i in range(n_subsets):
            subset = df.sample(n=subset_size, random_state=i)
            out_path = os.path.join(output_dir, f"subset_{i+1}.csv")
            subset.to_csv(out_path, index=False)
            print(f"subset_{i+1}.csv gespeichert unter {out_path}")
