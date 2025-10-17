import os
import pandas as pd
from pathlib import Path
import sys

# Importiere alle Module
from convertARFFtoCSV import initialize_datasets
from createBadgersDatasets import apply_noise_and_save
from kCenterGreedy import create_kcenter_subsets
from randomSubset import create_random_subsets
from TabPFNonSubsets import calc_tabpfn
from testDeepCore import run_deepcore_subset
from testTabPfnImp import runIMLFunctions

subset_size = 500 

def main(datasetname):
    if datasetname == "all":
        new_dataset_names = initialize_datasets(Path(os.getcwd()).parent)
    else:
        new_dataset_names = [datasetname]
        #ensure the dataset folder exists
        dataset_folder = Path(os.getcwd()).parent / "datasets" / datasetname
        if not dataset_folder.exists():
            print(f"Dataset folder {dataset_folder} does not exist. Exiting.")
            return
    print(f"Neue Datensätze erkannt: {new_dataset_names}")
    
    # Falls du nur mit neuen weiterarbeiten willst:
    for name in new_dataset_names:
        #print(f"Verarbeite Datensatz: {name}")
        #apply_noise_and_save(name, Path(os.getcwd()).parent)

        #print(f"Erstelle Random Subsets für: {name}")
        #create_random_subsets(name, Path(os.getcwd()).parent, n_subsets=3, subset_size=subset_size)
        
        #print(f"Erstelle k-Center Subsets für: {name}")
        #create_kcenter_subsets(name, Path(os.getcwd()).parent, k=subset_size)
        
        #print(f"Erstelle DeepCore Subsets für: {name}")
        #run_deepcore_subset(name, Path(os.getcwd()).parent)

        runIMLFunctions(name, Path(os.getcwd()).parent)

        #print(f"Führe TabPFN auf kCenter für: {name} aus")
        #calc_tabpfn(name, method="kCenter")

if __name__ == "__main__":
    #get arguments from command line
    args = sys.argv[1:]
    if args:
        main(args[0])
    else:
        main("all")