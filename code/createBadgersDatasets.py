###########################################
# Script to create datasets with various types of noise using the Badgers library
# last updated: 1.9.25
# last state: all the error level are final 
# notes: only for binary classification? did i look at regression?
############################################

import pandas as pd
from scipy.io import arff
from badgers.generators.tabular_data.noise import GaussianNoiseGenerator
from badgers.generators.tabular_data.missingness import MissingCompletelyAtRandom
from badgers.generators.tabular_data.outliers import HypersphereSamplingGenerator

from numpy.random import default_rng
from pathlib import Path

import sys
#gehe eine ebene runter
sys.path.append("..") 
from external.badgers.badgers.generators.tabular_data.imbalance import RandomUniqueBinaryClassesGenerator

def MyGaussianNoiseGenerator(X, y, noise_levels, target_col, output_dir, prefix):
    for std in noise_levels:
        transformer = GaussianNoiseGenerator(random_generator=default_rng(42))
        Xt, yt = transformer.generate(X.copy(), y, noise_std=std)
        df_noisy = Xt.copy()
        df_noisy[target_col] = yt
        filename = output_dir / f"{prefix}_GaussianFeatureNoise_std{int(std*100):02}.csv"
        df_noisy.to_csv(filename, index=False)

def MyMissingCompletelyAtRandom(X, y, missing_values, target_col, output_dir, prefix):
    for percentage in missing_values:
        transformer = MissingCompletelyAtRandom(random_generator=default_rng(42))
        Xm, _ = transformer.generate(X.copy(), y, percentage_missing=percentage)
        df_missing = Xm.copy()
        df_missing[target_col] = y
        filename = output_dir / f"{prefix}_MissingCompletely_std{int(percentage*100):02}.csv"
        df_missing.to_csv(filename, index=False)

def MyOutlierGenerator(X, y, percent_outliers, target_col, output_dir, prefix):
    for num in percent_outliers:
        number_outliers = int(num * len(X))
        transformer = HypersphereSamplingGenerator(random_generator=default_rng(42))
        Xo, Y0 = transformer.generate(X.copy(), y=y, n_outliers=number_outliers)
        df_outliers = pd.DataFrame(Xo, columns=X.columns)
        df_outliers[target_col] = y
        X[target_col] = y
        df_outliers = pd.concat([X, df_outliers], ignore_index=True)
        filename = output_dir / f"{prefix}_Outlier_std{int(num*100):02}.csv"
        df_outliers.to_csv(filename, index=False)

# classimbalancedness

def MyRandomSamplingClassesGenerator(X, y, imbalance_levels, target_col, output_dir, prefix):
    if len(y.unique()) == 2:
        #see what class has more instances
        class_counts = y.value_counts()
        for imbalance in imbalance_levels:
            if class_counts.iloc[0] > class_counts.iloc[1]:
                proportion_classes = {y.unique()[0]:imbalance[0], y.unique()[1]:imbalance[1]}
            else:
                proportion_classes = {y.unique()[0]:imbalance[1], y.unique()[1]:imbalance[0]}
            transformer = RandomUniqueBinaryClassesGenerator(random_generator=default_rng(42))
            X_imbalanced, y_imbalanced = transformer.generate(X.copy(), y, proportion_classes=proportion_classes)
            df_imbalanced = X_imbalanced.copy()
            df_imbalanced[target_col] = y_imbalanced
            filename = output_dir / f"{prefix}_ClassImbalancedness_ratio{int((imbalance[1]/imbalance[0]*100)):02}.csv"
            df_imbalanced.to_csv(filename, index=False)
    else:
        print(f"Skipping {prefix} as it does not have exactly two classes for imbalance generation.")

def apply_noise_and_save(dataset_name: str, rootPath: Path):
    # Parameter
    ALL_PERCANTAGES = [0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]
    NOISE_LEVELS = [0.1, 0.3, 0.5]
    MISSING_VALUES = [0.1, 0.3, 0.5]
    OUTLIER_PERCENT = [0.2, 0.6, 1]
    IMBALANCE_LEVELS = [[0.5, 0.5], [0.67, 0.33], [0.95, 0.05]]

    output_dir = rootPath / "datasets" / dataset_name
    dataset_path = output_dir / f"{dataset_name}_original.csv"
    print(f"Processing dataset: {dataset_path}")
    if not dataset_path.exists():
        print(f"Original CSV für {dataset_name} nicht gefunden: {dataset_path}")
        return

    df = pd.read_csv(dataset_path)
    target_col = df.columns[-1]
    X = df.drop(columns=target_col)
    y = df[target_col]
    prefix = f"{dataset_name}"

    # Beispiel: nur eine der Transformationen aktiviert
    # Du kannst beliebige davon aktivieren/deaktivieren

    MyGaussianNoiseGenerator(X, y, NOISE_LEVELS, target_col, output_dir, prefix)
    MyMissingCompletelyAtRandom(X, y, MISSING_VALUES, target_col, output_dir, prefix)
    MyOutlierGenerator(X, y, OUTLIER_PERCENT, target_col, output_dir, prefix)
    MyRandomSamplingClassesGenerator(X, y, IMBALANCE_LEVELS, target_col, output_dir, prefix)