import os
#import pandas as pd
from pathlib import Path
import sys

BaseDir = Path(os.getcwd()).parent
sys.path.append(str(BaseDir))

# Importiere alle Module
from code.convertARFFtoCSV import initialize_datasets
from code.createBadgersDatasets import apply_noise_and_save
from code.kCenterGreedy import create_kcenter_subsets
from code.randomSubset import create_random_subsets
from code.TabPFNonSubsets import calc_tabpfn
from code.TabPFNIMLSets import calc_tabpfn_iml
from code.testDeepCore import run_deepcore_subset
from code.testTabPfnImp import runIMLFunctions
from code.plotAllLevel import plotAllLevels

subset_size = 10000

IMLFunktionen = {
    "ALE": False,
    "Conformal": False,
    "Counterfactuals": False,
    "Data_Shapley": False,
    "DCA": False,
    "ICE_PD": False,
    "SHAP_Package": False,
    "SHAP": False,
    "LOCO": False,
    "Sensitivity": False,
    "OptimalSubset_roc": True,
    "OptimalSubset_b_acc": False,
    "OptimalSubset_f1": False,
    "OptimalSubset_ece": False, 
    "OptimalSubset_mce": False
}

def main(datasetname):
    if datasetname == "all":
        new_dataset_names = initialize_datasets(Path(os.getcwd()))
    else:
        new_dataset_names = [datasetname]
        #ensure the dataset folder exists
        dataset_folder = Path(os.getcwd()) / "datasets" / datasetname
        if not dataset_folder.exists():
            print(f"Dataset folder {dataset_folder} does not exist. Exiting.")
            return
    print(f"Neue Datensätze erkannt: {new_dataset_names}")
    
    # Falls du nur mit neuen weiterarbeiten willst:
    for name in new_dataset_names:
        #print(f"Verarbeite Datensatz: {name}")
        #apply_noise_and_save(name, Path(os.getcwd()))

        #print(f"Erstelle Random Subsets für: {name}")
        #create_random_subsets(name, Path(os.getcwd()), n_subsets=10, subset_size=subset_size)
        
        #print(f"Erstelle k-Center Subsets für: {name}")
        #create_kcenter_subsets(name, Path(os.getcwd()), k=subset_size)
        
        #print(f"Erstelle DeepCore Subsets für: {name}")
        #run_deepcore_subset(name, Path(os.getcwd()))

        #runIMLFunctions(name, Path(os.getcwd()), IMLFunktionen)

        print(f"Führe TabPFN auf random für: {name} aus")
        calc_tabpfn(name, method="random")

        #print(f"Führe TabPFN auf LOCO für: {name} aus")
        #calc_tabpfn_iml(name, IMLFunktionen)

        print("plotAllLevel.py wird aufgerufen")
        plotAllLevels(name, IMLFunktionen)


if __name__ == "__main__":
    #get arguments from command line
    args = sys.argv[1:]
    if args:
        main(args[0])
    else:
        main("all")