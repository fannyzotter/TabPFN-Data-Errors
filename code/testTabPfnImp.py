import numpy as np
import pandas as pd

from pathlib import Path
import os

from external.tabpfn_iml.tabpfniml.datasets.datasets import ArrayData

from external.tabpfn_iml.tabpfniml.methods.ale import ALE
from external.tabpfn_iml.tabpfniml.methods.conformal_pred import Conformal_Prediction
from external.tabpfn_iml.tabpfniml.methods.counterfactuals import Counterfactuals
from external.tabpfn_iml.tabpfniml.methods.data_shapley import Data_Shapley
from external.tabpfn_iml.tabpfniml.methods.dca import DCA
from external.tabpfn_iml.tabpfniml.methods.ice_pd import ICE_PD
from external.tabpfn_iml.tabpfniml.methods.kernel_shap_package import SHAP_Package_Wrapper 
from external.tabpfn_iml.tabpfniml.methods.kernel_shap import SHAP
from external.tabpfn_iml.tabpfniml.methods.loco import LOCO
from external.tabpfn_iml.tabpfniml.methods.sensitivity import Sensitivity

N_ensemble_configurations=1
device="auto"

def runIMLFunctions(dataset_name, base_dir):
    
    print("In runIMLFunktions")

    # yes runs until boxplots no idea what they mean
    #print(f"Führe ALE auf: {dataset_name} aus")
    #tryAle(dataset_name, Path(os.getcwd()).parent)

    # maybe errors at Mapie lib thingy no idea what
    #print(f"Führe Conformal Prediction auf: {dataset_name} aus")
    #tryConformalPred(dataset_name, Path(os.getcwd()).parent)

    # maybe something in the fit arguments is missing idk what X_factuals?!?!
    #print(f"Führe Conformal Prediction auf: {dataset_name} aus")
    #tryCounterfactuals(dataset_name, Path(os.getcwd()).parent)

    # no does only work with openMl Id. my datasets do not have this
    #print(f"Führe Data Shapely auf: {dataset_name} aus")
    #tryDataShap(dataset_name, Path(os.getcwd()).parent)

    # yes plots something I do not realy understand, lightgbm and gradiant boosting errors partialy because of missing values and NANs
    #print(f"Führe DCA auf: {dataset_name} aus")
    #tryDCA(dataset_name, Path(os.getcwd()).parent)

    # yes? errors in plotting idk why pls help
    #print(f"Führe ICE_PD auf: {dataset_name} aus")
    #tryIcePd(dataset_name, base_dir)

    # yes? takes a long time idk what is happening 
    # something is happening but takes to long as of right now
    #print(f"Führe SHAP Package auf: {dataset_name} aus")
    #tryShapPackage(dataset_name, base_dir)

    # yes? works as of now but do not know what the plots say
    #print(f"Führe SHAP auf: {dataset_name} aus")
    #tryShap(dataset_name, base_dir)

    # yes plots take to long will try again later
    #print(f"Führe LOCO auf: {dataset_name} aus")
    #tryLOCO(dataset_name, Path(os.getcwd()).parent)

    # no as of now because of store_gradiants=True did not work look into storage of the X and X_train arrays there is a dimension missing
    #print(f"Führe Sensitivity auf: {dataset_name} aus")
    #trySensitivity(dataset_name, Path(os.getcwd()).parent)





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

        sens_obj = ALE(data=array_data, n_train=data_train, n_test=data_test, device=device, N_ensemble_configurations=N_ensemble_configurations)
        sens_obj.fit(compute_wrt_feature=True,
             compute_wrt_observation=True,
             loss_based=True,
             pred_based=True)
        sens_obj.boxplot()

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

def tryDataShap(dataset_name, base_dir):
    input_path = base_dir / "datasets" / dataset_name
    output_root = base_dir / "ImlDataSHAPSubsets"

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
        sens_obj = Data_Shapley(data=array_data, n_train=data_train, n_test=data_test, device=device, N_ensemble_configurations=N_ensemble_configurations)
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
    output_root = base_dir / "ImlDataSHAPSubsets"

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
        sens_obj = Data_Shapley(data=array_data, n_train=data_train, n_test=data_test, device=device, N_ensemble_configurations=N_ensemble_configurations)
        sens_obj.fit()
        sens_obj.plot_bar()


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
    output_root = base_dir / "ImlLOCOSubsets"

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
        sens_obj = LOCO(data=array_data, n_train=data_train, n_test=data_test, device=device, N_ensemble_configurations=N_ensemble_configurations)
        sens_obj.fit(compute_wrt_feature=False,
             compute_wrt_observation=True,
             loss_based=False,
             pred_based=True, 
             class_to_be_explained=1)
        sens_obj.boxplot(plot_pred_based=True, plot_wrt_observation=True)

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


