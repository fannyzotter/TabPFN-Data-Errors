#################################################
# Script der score je level  plotts 
# last updated: 1.9.25
# last state: working plots as is in final thesis
# notes: maybe text size in plots a bit small
#################################################

import pandas as pd
import matplotlib.pyplot as plt
import re
import os

def extract_type_level(path):
    path = path.replace(" ", "_").replace(".", "")
    match = re.search(r"_(Outlier|GaussianFeatureNoise|MissingCompletely|ClassImbalancedness)_(std|ratio)(\d+)", path)
    if match:
        return match.group(1), int(match.group(3))
    return None, None

def createPlotsForAllLevels(csv_file, output_dir):
    print(f"Lade CSV-Datei: {csv_file}")
    df = pd.read_csv(csv_file)

    # Zeilen mit fehlender Accuracy ignorieren
    df = df.dropna(subset=["balanced_acc"])

    # Neue Spalten mit Typ und Level erzeugen
    df[['type', 'level']] = df['subset_path'].apply(lambda p: pd.Series(extract_type_level(p)))

    # Relevante Scores
    score_cols = ['balanced_acc', 'f1', 'roc_auc']

    # Gruppiert nach Typ und Level den Mittelwert je Score berechnen
    grouped = df.groupby(['type', 'level'])[score_cols].mean().reset_index()

    # Plot für Outlier
    subset = grouped[grouped['type'] == 'Outlier']
    plt.figure(figsize=(10, 6))
    for col in score_cols:
        plt.plot(subset['level'].to_numpy()/2, subset[col].to_numpy(), marker='o', label=col)
    plt.title("Outlier: Scores vs. Level")
    plt.xlabel("Outlier Level (%)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/OutlierAllLevel.png")
    plt.close()

    # Plot für GaussianFeatureNoise
    subset = grouped[grouped['type'] == 'GaussianFeatureNoise']
    plt.figure(figsize=(10, 6))
    for col in score_cols:
        plt.plot(subset['level'].to_numpy(), subset[col].to_numpy(), marker='o', label=col)
    plt.title("Gaussian Feature Noise: Scores vs. Level")
    plt.xlabel("Noise Level (std %)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/FeatureNoiseAllLevel.png")
    plt.close()

    # Plot für MissingCompletely
    subset = grouped[grouped['type'] == 'MissingCompletely']
    plt.figure(figsize=(10, 6))
    for col in score_cols:
        plt.plot(subset['level'].to_numpy(), subset[col].to_numpy(), marker='o', label=col)
    plt.title("Missing Completely: Scores vs. Level")
    plt.xlabel("Missing Ratio (%)")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/MissingCompletelyAllLevel.png")
    plt.close()

    # Plot für ClassImbalancedness
    subset = grouped[grouped['type'] == 'ClassImbalancedness']
    plt.figure(figsize=(10, 6))
    for col in score_cols:
        plt.plot(subset['level'].to_numpy()/100, subset[col].to_numpy(), marker='o', label=col)
    plt.title("Class Imbalancedness: Scores vs. Level")
    plt.xlabel("Class Ratio")
    plt.ylabel("Score")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/ClassImbalancednessAllLevel.png")
    plt.close()
    
def plotAllLevels(dataset_name, methods):
    print("Starte Erstellung aller Level Plots")
    if methods is not str:
        for method, action in methods.items():
            if action:
                print(f"Erstelle Plots für alle Level von {dataset_name} mit Methode {method}")
                # CSV-Datei laden
                dir_path = "Iml" + method + "Subsets/" + dataset_name + "/"
                csv_file = dir_path + dataset_name + "Log2" + method + ".csv"
                createPlotsForAllLevels(csv_file, dir_path)
    elif methods is str:
        print(f"Erstelle Plots für alle Level von {dataset_name} mit Methode {methods}")
        dir_path = "../results/" + methods + "/"
        csv_file = dir_path + "Aus" + "Log" + ".csv"
        createPlotsForAllLevels(csv_file, dir_path)

    print("Fertig mit Erstellung aller Level Plots")