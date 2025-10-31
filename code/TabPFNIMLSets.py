import os
import time
import pandas as pd
import numpy as np

import multiprocessing as mp
mp.set_start_method("spawn", force=True)

from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics
from tabpfn import TabPFNClassifier

import signal

class TimeoutException(Exception):
    pass

def handler(signum, frame):
    raise TimeoutException()

signal.signal(signal.SIGALRM, handler)

# Parameter

test_size = 0.2
timeout_sec = 180  # Max. Laufzeit pro Subset in Sekunden

def logfile_path_iml(dataset_name, method):
    print(f"Logfile für {dataset_name} mit Methode {method} wird erstellt")
    dir_path = "Iml" + method + "Subsets/" + dataset_name + "/"
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

    #subsets = method + "Subsets"
    input_root = dir_path

    output_log = dir_path + dataset_name + "Log2" + method + ".csv"
    print(output_log)
    if not os.path.exists(output_log):
        with open(output_log, "w") as f:
            f.write("dataset,subset_path,subset_name,n_samples,n_features,balanced_acc,f1,roc_auc,inference_time_sec,status\n")
    print("Logfile wurde erstellt")
    return input_root, output_log


# Funktion für separaten Subprozess
def run_tabpfn(X_train, y_train, X_test):
    clf = TabPFNClassifier(device="cuda", n_estimators=4)
    start_time = time.time()
    clf.fit(X_train, y_train)
    emb = clf.get_embeddings(X_test, data_source="test")
    print("Embeddings shape:", emb.shape)
    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)
    duration = time.time() - start_time
    return y_pred, duration

def calc_tabpfn_iml(dataset_name, imlFunktionen):
    for method, active in imlFunktionen.items():
        if active:
            print(f"Starte TabPFN für {dataset_name} mit Methode {method}")
            # Logdatei mit Header erzeugen, falls nicht vorhanden
            input_root, output_file = logfile_path_iml(dataset_name, method)

            print("runing tabpfn")
            # Hauptloop über alle Subsets
            for dirpath, _, filenames in os.walk(input_root):
                # only go in directories that start with australian_Class
                print(f"Verarbeite Verzeichnis: {dirpath}")
                for file in filenames:
                    print(f"Verarbeite Datei: {file}")
                    if file.endswith("scores.csv"):
                        continue
                    if not file.endswith(".csv"):
                        continue
                    subset_path = os.path.join(dirpath, file)
                    subset_name = os.path.splitext(file)[0]
                    dataset = os.path.basename(os.path.dirname(dirpath))

                    try:
                        df = pd.read_csv(subset_path)
                    except Exception as e:
                        print(f"❌ Fehler beim Laden von {subset_path}: {e}")
                        continue
                    
                    if df.shape[0] < 50:
                        print(f"⚠️ {subset_path} hat zu wenig Zeilen – überspringe.")
                        continue
                    
                    # Feature/Target aufteilen
                    X = df.iloc[:, :-1].values
                    y = df.iloc[:, -1].values

                    X_train, X_test, y_train, y_test = train_test_split(
                        X, y, test_size=test_size, stratify=y if len(set(y)) > 1 else None
                    )

                    try:
                        signal.alarm(timeout_sec)  # Timeout setzen
                        y_pred, duration = run_tabpfn(X_train, y_train, X_test)
                        #acc = metrics.accuracy_score(y_test, y_pred)
                        balanced_acc = metrics.balanced_accuracy_score(y_test, y_pred)
                        #precision = metrics.precision_score(y_test, y_pred, average="binary")
                        #recall = metrics.recall_score(y_test, y_pred, average="binary")
                        f1 = metrics.f1_score(y_test, y_pred, average="binary")
                        roc_auc = metrics.roc_auc_score(y_test, y_pred)
                        #mcc = metrics.matthews_corrcoef(y_test, y_pred)
                        #kappa = metrics.cohen_kappa_score(y_test, y_pred)
                        status="ok"
                        signal.alarm(0)            # Timeout wieder ausschalten
                    except TimeoutException:
                        print(f"⏱️ Timeout bei {subset_path}")
                        status = "timeout"
                        balanced_acc = f1 = roc_auc = ""
                        duration = timeout_sec
                    with open(output_file, "a") as f:
                        f.write(f"{dataset},{subset_path},{subset_name},{df.shape[0]},{df.shape[1]-1},{balanced_acc},{f1},{roc_auc},{duration},{status}\n")