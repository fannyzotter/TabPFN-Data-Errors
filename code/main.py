import os
import pandas as pd
from pathlib import Path
import sys 

# Importiere alle Module
from convertARFFtoCSV import initialize_datasets
from createBadgersDatasets import apply_noise_and_save
#import TabPFNonSubsets
from kCenterGreedy import create_kcenter_subsets
 #import knnShapely
from randomSubset import create_random_subsets

def main():
    new_dataset_names = initialize_datasets(Path(os.getcwd()).parent)

    print(f"Neue Datensätze erkannt: {new_dataset_names}")
    
    # Falls du nur mit neuen weiterarbeiten willst:
    for name in new_dataset_names:
        #apply_noise_and_save(name, Path(os.getcwd()).parent)
        # create subsets
        #create_random_subsets(name, Path(os.getcwd()).parent, n_subsets=3, subset_size=500)

        create_kcenter_subsets(name, Path(os.getcwd()).parent, k=500)

if __name__ == "__main__":
    main()
