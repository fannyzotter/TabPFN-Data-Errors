import os
import pandas as pd
import random
from pathlib import Path

def create_random_subsets(dataset_name, base_path, n_subsets=10, subset_size=500):
    """
    Erstellt zufällige Subsets für einen gegebenen Datensatz
    
    Args:
        dataset_name (str): Name des Datensatzes
        base_path (Path): Basis-Pfad zum Projekt
        n_subsets (int): Anzahl der zu erstellenden Subsets
        subset_size (int): Größe jedes Subsets
    """
    input_path = base_path / "datasets" / dataset_name
    output_root = base_path / "randomSubsets"
    
    for filename in os.listdir(input_path):
        # Nur bestimmte Dateien verarbeiten (hier: Australian Datensätze)
        if not filename.endswith(".csv") or "original" in filename:
            continue
            
        print(f"Verarbeite Datei: {filename}")
        file_path = input_path / filename

        try:
            df = pd.read_csv(file_path)
        except Exception as e:
            print(f"Fehler beim Laden: {e}")
            continue

        # Zielordner vorbereiten
        base_name = filename.replace(" ", "_").replace(".", "").replace("csv", "")
        output_dir = output_root / dataset_name / base_name
        
        current_subset_size = subset_size
        if len(df) < subset_size:
            print(f"Datei hat nur {len(df)} Zeilen.")
            output_dir = Path(str(output_dir) + f"_{len(df)}rows")
            current_subset_size = len(df)

        # Verzeichnis erstellen (inkl. Elternverzeichnisse)
        output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Output-Verzeichnis erstellt: {output_dir}")

        # Erstelle Subsets
        for i in range(n_subsets):
            subset = df.sample(n=current_subset_size, random_state=i)
            out_path = output_dir / f"subset_{i+1}.csv"
            subset.to_csv(out_path, index=False)
            print(f"subset_{i+1}.csv gespeichert unter {out_path}")