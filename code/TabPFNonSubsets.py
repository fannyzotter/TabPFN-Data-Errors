import os
import time
import pandas as pd
import numpy as np
from multiprocessing import Process, Queue
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

from tabpfn import TabPFNClassifier

# Parameter
#input_root = "../randomSubsets/numerai28.6"
input_root = "../randomSubsets/PhishingWebsites"
logfile_path = "../results/log.csv"
test_size = 0.2
timeout_sec = 300  # Max. Laufzeit pro Subset in Sekunden

# Logdatei mit Header erzeugen, falls nicht vorhanden
if not os.path.exists(logfile_path):
    with open(logfile_path, "w") as f:
        f.write("dataset,subset_path,subset_name,n_samples,n_features,test_acc,inference_time_sec,status\n")

# Funktion für separaten Subprozess
def run_tabpfn(X_train, y_train, X_test, queue):
    clf = TabPFNClassifier(device="cuda")
    start_time = time.time()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    duration = time.time() - start_time
    queue.put((y_pred, duration))

# Hauptloop über alle Subsets
for dirpath, _, filenames in os.walk(input_root):
    for file in filenames:
        if not file.endswith("_1.csv"):
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

        queue = Queue()
        p = Process(target=run_tabpfn, args=(X_train, y_train, X_test, queue))
        p.start()
        p.join(timeout_sec)

        if p.is_alive():
            p.terminate()
            p.join()
            print(f"⏱️ Timeout bei {subset_path}")
            status = "timeout"
            acc = ""
            duration = timeout_sec
        elif not queue.empty():
            y_pred, duration = queue.get()
            acc = accuracy_score(y_test, y_pred)
            status = "ok"
            print(f"✅ {subset_name}: acc={acc:.4f}, time={duration:.2f}s")
        else:
            acc = ""
            duration = ""
            status = "error"
            print(f"⚠️ Fehler bei {subset_path}")

        with open(logfile_path, "a") as f:
            f.write(f"{dataset},{subset_path},{subset_name},{df.shape[0]},{df.shape[1]-1},{acc},{duration},{status}\n")
