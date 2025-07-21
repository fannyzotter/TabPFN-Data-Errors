import os
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# Parameter
input_root = "../datasets"  # Eingabe-Ordner mit CSV-Dateien
output_root = "../knnShapleySubsets"
k = 500  # Anzahl der Punkte im Subset
k_neighbors = 3  # für KNN-Klassifikation
test_ratio = 0.2  # Anteil der Daten als Testset für Bewertung

def knn_shapley(X, y, X_test, y_test, k_neighbors=3):
    scores = np.zeros(len(X))
    for i in range(len(X)):
        mask = np.ones(len(X), dtype=bool)
        mask[i] = False

        model_without = KNeighborsClassifier(n_neighbors=k_neighbors)
        model_without.fit(X[mask], y[mask])
        acc_without = accuracy_score(y_test, model_without.predict(X_test))

        model_with = KNeighborsClassifier(n_neighbors=k_neighbors)
        model_with.fit(X, y)
        acc_with = accuracy_score(y_test, model_with.predict(X_test))

        scores[i] = acc_with - acc_without
    return scores

# Erstelle Ausgabeordner falls nicht vorhanden
os.makedirs(output_root, exist_ok=True)

# Iteriere durch alle Datasets
for dataset_name in os.listdir(input_root):
    print(f"Verarbeite Dataset: {dataset_name}")
    dataset_path = os.path.join(input_root, dataset_name)

    # Nur Ordner berücksichtigen
    if not os.path.isdir(dataset_path):
        continue
    if not dataset_name.startswith("Australian"):
        continue

    # Erstelle entsprechenden Output-Unterordner
    dataset_out_path = os.path.join(output_root, dataset_name.replace(" ", "_").replace(".", ""))
    os.makedirs(dataset_out_path, exist_ok=True)

    # Iteriere über alle CSV-Dateien im Dataset-Ordner
    for filename in os.listdir(dataset_path):
        if not filename.startswith("Australian") or not filename.endswith(".csv"):
            continue

        file_path = os.path.join(dataset_path, filename)
        out_file = os.path.join(dataset_out_path, filename.replace(".csv", f"_knnshapley{k}.csv").replace(".", ""))

        try:
            df = pd.read_csv(file_path)

            # Features und Label trennen (Label = letzte Spalte)
            X = df.iloc[:, :-1].values
            y = df.iloc[:, -1].values

            # NaNs behandeln
            X = pd.DataFrame(X).fillna(pd.DataFrame(X).mean()).values

            # Normalisieren
            X_scaled = StandardScaler().fit_transform(X)

            # Split in Train + Test (einfach zufällig)
            n_total = len(X_scaled)
            n_test = int(test_ratio * n_total)
            indices = np.random.permutation(n_total)
            test_idx = indices[:n_test]
            train_idx = indices[n_test:]

            X_train, y_train = X_scaled[train_idx], y[train_idx]
            X_test, y_test = X_scaled[test_idx], y[test_idx]

            # Shapley-Werte berechnen
            shapley_scores = knn_shapley(X_train, y_train, X_test, y_test, k_neighbors=k_neighbors)

            # Top‑k auswählen
            top_k_idx = np.argsort(shapley_scores)[-k:]
            selected_idx = train_idx[top_k_idx]  # auf globale Indizes abbilden
            df_subset = df.iloc[selected_idx]

            # Speichern
            df_subset.to_csv(out_file, index=False)
            print(f"✅ {filename} → KNN-Shapley-Subset gespeichert ({len(df_subset)} Punkte)")

        except Exception as e:
            print(f"⚠️ Fehler bei {file_path}: {e}")
