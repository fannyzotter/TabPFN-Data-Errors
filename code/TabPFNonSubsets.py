import os
import time
import pandas as pd
import numpy as np
from tabpfn import TabPFNClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

# Parameter
input_root = "../subsets"                   # Ordner mit allen Subsets
logfile_path = "../results/log.csv"         # Zentrale Logdatei
test_size = 0.2                             # 80/20 Split für Evaluation

# Erstelle Log-Datei mit Header (falls sie nicht existiert)
if not os.path.exists(logfile_path):
    with open(logfile_path, "w") as f:
        f.write("dataset,subset_path,subset_name,n_samples,n_features,test_acc,inference_time_sec\n")

# Durchlaufe alle .csv-Dateien in allen Unterordnern
for dirpath, _, filenames in os.walk(input_root):
    for file in filenames:
        if not file.endswith(".csv"):
            continue

        subset_path = os.path.join(dirpath, file)
        subset_name = os.path.splitext(file)[0]
        dataset = os.path.basename(os.path.dirname(dirpath))

        try:
            df = pd.read_csv(subset_path)
        except Exception as e:
            print(f"Fehler beim Laden von {subset_path}: {e}")
            continue

        if df.shape[0] < 50:
            print(f"{subset_path} hat zu wenig Zeilen – überspringe.")
            continue

        # Features / Target trennen (Zielspalte = letzte Spalte)
        X = df.iloc[:, :-1].values
        y = df.iloc[:, -1].values

        # Split in Train/Test
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y if len(set(y)) > 1 else None)

        # TabPFN trainieren + Zeit messen
        clf = TabPFNClassifier(device="cuda")  # oder "cuda" falls GPU vorhanden
        start_time = time.time()
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        duration = time.time() - start_time

        acc = accuracy_score(y_test, y_pred)

        # Log schreiben
        with open(logfile_path, "a") as f:
            f.write(f"{dataset},{subset_path},{subset_name},{df.shape[0]},{df.shape[1]-1},{acc:.4f},{duration:.2f}\n")

        print(f"{subset_name}: acc={acc:.4f}, time={duration:.2f}s")
