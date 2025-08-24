from pathlib import Path
import arff
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
                dataset = arff.load(f)
            df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])
            # Als .csv speichern
            csv_path = dataset_folder / f"{name}_original.csv"
            df.to_csv(csv_path, index=False)
        new_datasets.append(name)

    return new_datasets