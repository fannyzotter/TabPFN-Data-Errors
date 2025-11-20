from pathlib import Path
from scipy.io import arff
import pandas as pd

def initialize_datasets(rootpath: Path):
    #combine rootpath and arff_path
    arff_path = rootpath.joinpath("datasets/arffDatasets")
    arff_files = list(arff_path.glob("*.arff"))
    new_datasets = []
    print(f"Found {len(arff_files)} ARFF files in {arff_path}")
    for arff_file in arff_files:
        name = arff_file.stem
        dataset_folder = rootpath / "datasets" / name

        if not dataset_folder.exists():
            print(f"Creating dataset folder: {dataset_folder}")
            dataset_folder.mkdir()
            # .arff-Datei laden
            with open(arff_file) as f:
                dataset = arff.loadarff(f)
                print(f"Loaded ARFF file: {arff_file}")
                print(dataset.count)
            df = pd.DataFrame(dataset[0])
            df = df.applymap(lambda x: x.decode("utf-8") if isinstance(x, bytes) else x)
            df = df.replace({"True": 1, "False": 0})
            try:
                df = df.astype(float)
            except ValueError:
                print(f"⚠️ Warnung: Einige Werte in {name} konnten nicht in float konvertiert werden.")
                print(ValueError.args)

            print(f"Converted ARFF to DataFrame with shape: {df.shape}")
            print(df.head())

            # Als .csv speichern
            csv_path = dataset_folder / f"{name}_original.csv"
            df.to_csv(csv_path, index=False)
            new_datasets.append(name)

    return new_datasets