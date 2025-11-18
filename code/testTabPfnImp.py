import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import re

from pathlib import Path
import os
import sys 
sys.path.append(str(Path(os.getcwd()).parent.parent))
print(sys.path)#.append(str(Path(os.getcwd()).parent.parent)))


from external.tabpfn_iml.tabpfniml.datasets.datasets import ArrayData

#from external.tabpfn_iml.tabpfniml.methods.ale import ALE
from external.tabpfn_iml.tabpfniml.methods.conformal_pred import Conformal_Prediction
from external.tabpfn_iml.tabpfniml.methods.counterfactuals import Counterfactuals
from external.tabpfn_iml.tabpfniml.methods.data_shapley import Data_Shapley
from external.tabpfn_iml.tabpfniml.methods.dca import DCA
from external.tabpfn_iml.tabpfniml.methods.ice_pd import ICE_PD
from external.tabpfn_iml.tabpfniml.methods.kernel_shap_package import SHAP_Package_Wrapper 
from external.tabpfn_iml.tabpfniml.methods.kernel_shap import SHAP
from external.tabpfn_iml.tabpfniml.methods.loco import LOCO
from external.tabpfn_iml.tabpfniml.methods.sensitivity import Sensitivity
from external.tabpfn_iml.tabpfniml.methods.interpret import TabPFN_Interpret


N_ensemble_configurations=1
device="cuda"


def runIMLFunctions(dataset_name, base_dir, IMLFunktionen):
    
    print("In runIMLFunktions")


    if IMLFunktionen.get("ALE", True):
        print(f"Führe ALE auf: {dataset_name} aus")
        #tryAle(dataset_name, base_dir) #yes runs until boxplots no idea what they mean
    
    if IMLFunktionen.get("Conformal", True):
        print(f"Führe Conformal Prediction auf: {dataset_name} aus")
        tryConformalPred(dataset_name, base_dir) #maybe errors at Mapie lib thingy no idea what
    
    if IMLFunktionen.get("Counterfactuals", True):
        print(f"Führe Counterfactuals auf: {dataset_name} aus")
        tryCounterfactuals(dataset_name, base_dir) #maybe something in the fit arguments is missing idk what X_factuals?!?!
    
    if IMLFunktionen.get("Data_Shapley", True): # prio 1 
        print(f"Führe Data Shapely auf: {dataset_name} aus")
        tryDataShap(dataset_name, base_dir) #no does only work with openMl Id. my datasets do not have this
    
    if IMLFunktionen.get("DCA", True):
        print(f"Führe DCA auf: {dataset_name} aus")
        tryDCA(dataset_name, base_dir) #yes plots something I do not realy understand, lightgbm and gradiant boosting errors partialy because of missing values and NANs
    
    if IMLFunktionen.get("ICE_PD", True):
        print(f"Führe ICE_PD auf: {dataset_name} aus")
        tryIcePd(dataset_name, base_dir) #yes? errors in plotting idk why pls help
    
    if IMLFunktionen.get("SHAP_Package", True): 
        print(f"Führe SHAP Package auf: {dataset_name} aus")
        tryShapPackage(dataset_name, base_dir) #yes? takes a long time idk what is happening
    
    if IMLFunktionen.get("SHAP", True):
        print(f"Führe SHAP auf: {dataset_name} aus")
        tryShap(dataset_name, base_dir) #yes? works as of now but do not know what the plots say
    
    if IMLFunktionen.get("LOCO", True): 
        print(f"Führe LOO auf: {dataset_name} aus")
        tryLOCO(dataset_name, base_dir) #yes but no
    
    if IMLFunktionen.get("Sensitivity", True):
        print(f"Führe Sensitivity auf: {dataset_name} aus")
        trySensitivity(dataset_name, base_dir) #no as of now because of store_gradiants=True did not work look into storage of the X and X_train arrays there is a dimension missing

    if IMLFunktionen.get("OptimalSubset_roc", True):
        print(f"Führe Optimal Subset ROC auf: {dataset_name} aus")
        tryOptimalSubset_roc(dataset_name, base_dir)



def tryAle(dataset_name, base_dir):
    input_path = base_dir / "datasets" / dataset_name
    output_root = base_dir / "ImlAleSubsets"


    for filename in os.listdir(input_path):
        print(f"Verarbeite Datei: {filename}")
        file_path = input_path / filename
        file_data = pd.read_csv(file_path)

        X = file_data.iloc[:, :-1]
        y = file_data.iloc[:, -1]

        #create ArrayData object
        array_data = ArrayData(dataset_name, X, y, feature_names=X.columns, categorical_features_idx=[])

        data_train = round(0.8 * array_data.X_df.shape[0])
        data_test = round(0.2 * array_data.X_df.shape[0])

        #sens_obj = ALE(data=array_data, n_train=data_train, n_test=data_test, device=device, N_ensemble_configurations=N_ensemble_configurations)
        #sens_obj.fit(compute_wrt_feature=True,
        #     compute_wrt_observation=True,
        #     loss_based=True,
        #     pred_based=True)
        #sens_obj.boxplot()

def tryConformalPred(dataset_name, base_dir):
    input_path = base_dir / "datasets" / dataset_name
    output_root = base_dir / "ImlConformlPredSubsets"

    for filename in os.listdir(input_path):
        print(f"Verarbeite Datei: {filename}")
        file_path = input_path / filename
        file_data = pd.read_csv(file_path)

        X = file_data.iloc[:, :-1]
        y = file_data.iloc[:, -1]

        #create ArrayData object
        array_data = ArrayData(dataset_name, X, y, feature_names=X.columns, categorical_features_idx=[])

        data_train = round(0.8 * array_data.X_df.shape[0])
        data_test = round(0.2 * array_data.X_df.shape[0])
        class_to_predict = len(file_data.columns) - 1
        print(f"Class to predict: {class_to_predict}")
        sens_obj = Conformal_Prediction(data=array_data, n_train=data_train, n_test=data_test, device=device, N_ensemble_configurations=N_ensemble_configurations)
        sens_obj.fit()
        sens_obj.plot_bar()

def tryCounterfactuals(dataset_name, base_dir):
    input_path = base_dir / "datasets" / dataset_name
    output_root = base_dir / "ImlCounterfactualsSubsets"

    for filename in os.listdir(input_path):
        print(f"Verarbeite Datei: {filename}")
        file_path = input_path / filename
        file_data = pd.read_csv(file_path)

        X = file_data.iloc[:, :-1]
        y = file_data.iloc[:, -1]

        #create ArrayData object
        array_data = ArrayData(dataset_name, X, y, feature_names=X.columns, categorical_features_idx=[])

        data_train = round(0.8 * array_data.X_df.shape[0])
        data_test = round(0.2 * array_data.X_df.shape[0])
        class_to_predict = len(file_data.columns) - 1
        print(f"Class to predict: {class_to_predict}")
        sens_obj = Counterfactuals(data=array_data, n_train=data_train, n_test=data_test, device=device, N_ensemble_configurations=N_ensemble_configurations)
        sens_obj.fit()
        sens_obj.plot_bar()

def tryDCA(dataset_name, base_dir):
    input_path = base_dir / "datasets" / dataset_name
    output_root = base_dir / "ImlDCASubsets"

    for filename in os.listdir(input_path):
        print(f"Verarbeite Datei: {filename}")
        file_path = input_path / filename
        file_data = pd.read_csv(file_path)

        X = file_data.iloc[:, :-1]
        y = file_data.iloc[:, -1]

        #create ArrayData object
        array_data = ArrayData(dataset_name, X, y, feature_names=X.columns, categorical_features_idx=[])

        data_train = round(0.8 * array_data.X_df.shape[0])
        data_test = round(0.2 * array_data.X_df.shape[0])
        class_to_predict = len(file_data.columns) - 1
        print(f"Class to predict: {class_to_predict}")
        sens_obj = DCA(data=array_data, n_train=data_train, n_test=data_test, device=device, N_ensemble_configurations=N_ensemble_configurations)
        sens_obj.fit(random_forest=True, gradient_boosting=True)
        sens_obj.plot(predictors=["Random Forest", "Gradient Boosting"])

def tryIcePd(dataset_name, base_dir):
    input_path = base_dir / "datasets" / dataset_name
    output_root = base_dir / "ImlIcePdSubsets"

    for filename in os.listdir(input_path):
        print(f"Verarbeite Datei: {filename}")
        file_path = input_path / filename
        file_data = pd.read_csv(file_path)

        X = file_data.iloc[:, :-1]
        y = file_data.iloc[:, -1]

        #create ArrayData object
        array_data = ArrayData(dataset_name, X, y, feature_names=X.columns, categorical_features_idx=[])

        data_train = round(0.8 * array_data.X_df.shape[0])
        data_test = round(0.2 * array_data.X_df.shape[0])
        class_to_predict = len(file_data.columns) - 1
        print(f"Class to predict: {class_to_predict}")
        sens_obj = ICE_PD(data=array_data, n_train=data_train, n_test=data_test, device=device, N_ensemble_configurations=N_ensemble_configurations)
        sens_obj.fit()
        sens_obj.plot()

def tryDataShap(dataset_name, base_dir):
    input_path = base_dir / "datasets" / dataset_name
    output_root = base_dir / "ImlData_ShapleySubsets" / dataset_name
    if not output_root.exists():
        output_root.mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_path):
        try:
            print(f"Verarbeite Datei: {filename}")
            file_path = input_path / filename
            file_data = pd.read_csv(file_path)

            basename = Path(filename).stem

            # check if file with basename as start of name exists 
            # maybe create regex to check because there is a part with subset size in the name
            regex = re.compile(rf"^{basename}_\d+_samples_subset\.csv$")
            if any(regex.match(f) for f in os.listdir(output_root)):
                print("skipping file ", basename)
                continue
            X = file_data.iloc[:, :-1]
            y = file_data.iloc[:, -1]

            #create ArrayData object
            array_data = ArrayData(dataset_name, X, y, feature_names=X.columns, categorical_features_idx=[])

            data_train = round(0.9 * array_data.X_df.shape[0])
            data_test = round(0.1 * array_data.X_df.shape[0])
            class_to_predict = len(file_data.columns) - 1
            subset_size = int(min(500, data_train))
            print(f"gesuchte Teilmenge: {subset_size} Beispiele")
            print(f"Anzahl Trainingsbeispiele: {data_train} Beispiele")
            print(f"Class to predict: {class_to_predict}")
            mfactor = int(data_train / subset_size) + 1
            print(f"Gesuchter M-Faktor: {mfactor}, {1/mfactor}% der Trainingsdaten")
            sens_obj = Data_Shapley(data=array_data, n_train=data_train, n_test=data_test, device=device, N_ensemble_configurations=N_ensemble_configurations)
            sens_obj.fit(M_factor=1/mfactor, tPFN_train_max=subset_size)
            indices = sens_obj.get_optimized_context()
            #save subset

            subset_x = array_data.X_df.iloc[indices]
            subset_y = array_data.y_df.iloc[indices]
            subset = pd.concat([subset_x, subset_y], axis=1)

            subset.to_csv(output_root / f"{basename}_{subset_size}_samples_subset.csv", index=False)
        except Exception as e:
            print(f"Fehler bei der Verarbeitung von {filename}: {e}")

def tryShap(dataset_name, base_dir):
    input_path = base_dir / "datasets" / dataset_name
    output_root = base_dir / "ImlSHAPSubsets"

    for filename in os.listdir(input_path):
        print(f"Verarbeite Datei: {filename}")
        file_path = input_path / filename
        file_data = pd.read_csv(file_path)

        X = file_data.iloc[:, :-1]
        y = file_data.iloc[:, -1]

        #create ArrayData object
        array_data = ArrayData(dataset_name, X, y, feature_names=X.columns, categorical_features_idx=[])

        data_train = round(0.8 * array_data.X_df.shape[0])
        data_test = round(0.2 * array_data.X_df.shape[0])
        class_to_predict = len(file_data.columns) - 1
        print(f"Class to predict: {class_to_predict}")
        sens_obj = SHAP(data=array_data, n_train=data_train, n_test=data_test, device=device, N_ensemble_configurations=N_ensemble_configurations)
        sens_obj.fit()
        sens_obj.plot_bar()

def tryShapPackage(dataset_name, base_dir):
    input_path = base_dir / "datasets" / dataset_name
    output_root = base_dir / "ImlSHAPPackageSubsets"

    for filename in os.listdir(input_path):
        print(f"Verarbeite Datei: {filename}")
        file_path = input_path / filename
        file_data = pd.read_csv(file_path)

        X = file_data.iloc[:, :-1]
        y = file_data.iloc[:, -1]

        #create ArrayData object
        array_data = ArrayData(dataset_name, X, y, feature_names=X.columns, categorical_features_idx=[])

        data_train = round(0.8 * array_data.X_df.shape[0])
        data_test = round(0.2 * array_data.X_df.shape[0])
        class_to_predict = len(file_data.columns) - 1
        print(f"Class to predict: {class_to_predict}")
        sens_obj = SHAP_Package_Wrapper(data=array_data, n_train=data_train, n_test=data_test, device=device, N_ensemble_configurations=N_ensemble_configurations)
        sens_obj.fit()
        sens_obj.plot_bar()

def tryLOCO(dataset_name, base_dir):
    input_path = base_dir / "datasets" / dataset_name
    Loco_output_root = base_dir / "ImlLOCOSubsets"
    Loco_output_root.mkdir(parents=True, exist_ok=True)

    for filename in os.listdir(input_path):
        print(f"Verarbeite Datei: {filename}")
        file_path = input_path / filename
        file_data = pd.read_csv(file_path)

        output_root = Loco_output_root / dataset_name
        output_root.mkdir(parents=True, exist_ok=True)

        # 5) Dateien speichern
        basename = Path(filename).stem
        scores_path  = output_root / f"{basename}_OE_scores.csv"
        
        #skip file if it is in output directory
        if (scores_path.exists()):
            print("skipping file ", basename)
        
        else:
            X = file_data.iloc[:, :-1]
            y = file_data.iloc[:, -1]

            #create ArrayData object
            array_data = ArrayData(dataset_name, X, y, feature_names=X.columns, categorical_features_idx=[])

            data_train = round(0.9 * array_data.X_df.shape[0])
            data_test = round(0.1 * array_data.X_df.shape[0])
            class_to_predict = len(file_data.columns) - 1
            print(f"Class to predict: {class_to_predict}")
            loo = LOCO(data=array_data, n_train=min(1024,data_train), n_test=data_test, device=device, N_ensemble_configurations=N_ensemble_configurations)
            loo.fit(compute_wrt_feature=False,
                compute_wrt_observation=True,
                loss_based=True, # calc importance
                pred_based=True, # calc effect
                n_train_relevance = min(1024,data_train),
                class_to_be_explained=1)
                    # ---- nach loo.fit(...) einfügen ----

            # 1) OE-Scores abrufen (global, d.h. pro Trainingspunkt)
            oe_scores = loo.get_OE(local=False)  # pd.Series: Index = Trainingspunkte, Wert = Einflussstärke

            # 2) Trainingsdaten-Indizes direkt aus dem Modell
            train_indices = np.arange(len(loo.X_train))
            print(f"Anzahl Trainingspunkte: {len(train_indices)}")
            print(f"Anzahl OE-Scores: {len(oe_scores)}")

            # 3) Score-Tabelle erstellen
            scores = pd.DataFrame({
                "train_idx": train_indices,
                "oe_score": oe_scores.values
            })

            # 4) Sortieren und Core-Set wählen
            pruning_ratio = 0.8          # 80% der Trainingspunkte
            context_cap = 500           # subset size limit
            k_by_ratio = int(len(train_indices) * pruning_ratio)
            k = min(k_by_ratio, context_cap)

            scores_sorted = scores.sort_values("oe_score", ascending=False, ignore_index=True)
            coreset_idx = scores_sorted.loc[:k-1, "train_idx"].to_numpy()

            subset_path  = output_root / f"{basename}_CoreSet_{k}.csv"
            meta_path    = output_root / f"{dataset_name}_meta.json"

            # a) vollständige Score-Tabelle
            scores_sorted.to_csv(scores_path, index=False)

            # b) Core-Set-Datensatz (Top-k Punkte aus Trainingsdaten)
            coreset_df = file_data.iloc[coreset_idx]
            coreset_df.to_csv(subset_path, index=False)

            # c) Metadaten

            meta = {
                "dataset": dataset_name,
                "source_file": str(file_path),
                "n_total": int(len(file_data)),
                "n_train": int(len(loo.X_train)) if hasattr(loo, "X_train") else None,
                "n_test": int(len(loo.X_test)) if hasattr(loo, "X_test") else None,
                "params_fit": {
                    "compute_wrt_feature": False,
                    "compute_wrt_observation": True,
                    "loss_based": False,
                    "pred_based": True,
                    "n_train_relevance": data_train,
                    "class_to_be_explained": 1,
                },
                "device": str(loo.device) if hasattr(loo, "device") else "unknown",
                "N_ensemble_configurations": int(N_ensemble_configurations),
            }

            try:
                with open(meta_path, "r") as f:
                    # add new information to existing json data
                    all_meta = json.load(f)
                    if not isinstance(all_meta, list):
                        all_meta = [all_meta]
            except Exception:
                all_meta = []

            all_meta.append(meta)
            print(f"[LOO] {basename}: Core-Set mit {k} Punkten gespeichert → {subset_path}")


    with open(meta_path, "w") as f:
        json.dump(all_meta, f, indent=2)

def trySensitivity(dataset_name, base_dir):
    input_path = base_dir / "datasets" / dataset_name
    output_root = base_dir / "ImlSensitivitySubsets"

    for filename in os.listdir(input_path):
        print(f"Verarbeite Datei: {filename}")
        file_path = input_path / filename
        file_data = pd.read_csv(file_path)

        X = file_data.iloc[:, :-1]
        y = file_data.iloc[:, -1]

        #create ArrayData object
        array_data = ArrayData(dataset_name, X, y, feature_names=X.columns, categorical_features_idx=[])

        data_train = round(0.8 * array_data.X_df.shape[0])
        data_test = round(0.2 * array_data.X_df.shape[0])
        class_to_predict = len(file_data.columns) - 1
        print(f"Class to predict: {class_to_predict}")
        sens_obj = Sensitivity(data=array_data, n_train=data_train, n_test=data_test, device=device, N_ensemble_configurations=N_ensemble_configurations)
        sens_obj.fit(compute_wrt_feature=True,
             compute_wrt_observation=True,
             loss_based=True,
             pred_based=True, 
             class_to_be_explained=1)
        sens_obj.boxplot()


def tryOptimalSubset_roc(dataset_name, base_dir):
    input_path = base_dir / "datasets" / dataset_name
    output_root = base_dir / "ImlOptimalSubsetRoc"

    for filename in os.listdir(input_path):
        print(f"Verarbeite Datei: {filename}")
        file_path = input_path / filename
        file_data = pd.read_csv(file_path)

        X = file_data.iloc[:, :-1]
        y = file_data.iloc[:, -1]

        #create ArrayData object
        array_data = ArrayData(dataset_name, X, y, feature_names=X.columns, categorical_features_idx=[])

        data_train = round(0.8 * array_data.X_df.shape[0])
        data_test = round(0.2 * array_data.X_df.shape[0])
        class_to_predict = len(file_data.columns) - 1
        print(f"Class to predict: {class_to_predict}")

        loo = LOCO(data=array_data, n_train=data_train, n_test=data_test, device=device, N_ensemble_configurations=N_ensemble_configurations)
        values = loo.fit_optimal_subset(metric="roc")
        print(values)