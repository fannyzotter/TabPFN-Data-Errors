import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from tabpfn import TabPFNClassifier
from tabpfn_extensions.post_hoc_ensembles.sklearn_interface import AutoTabPFNClassifier, AutoTabPFNRegressor

# CSV-Datei laden
df = pd.read_csv("archive/XAI_Drilling_Dataset.csv")

# Relevante Features und Zielvariable auswählen
features = ["Cutting speed vc [m/min]", "Spindle speed n [1/min]", "Feed f [mm/rev]", "Feed rate vf [mm/min]", "Power Pc [kW]", "Cooling [%]"]
target = "Main Failure"

X = df[features].values
y = df[target].values

if len(X) > 5000:
    X, y = X[:5000], y[:5000]

# Datensatz aufteilen
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Daten skalieren
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# TabPFN-Modell initialisieren und trainieren
clf = TabPFNClassifier(device='cpu', ignore_pretraining_limits=True)  # 'cuda' für GPU-Unterstützung
clf.fit(X_train, y_train)

# Vorhersagen treffen
y_pred = clf.predict(X_test)

# Ergebnisse anzeigen
print("Vorhersagen:", y_pred)

# Accuracy Measures berechnen
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1-Score: {f1:.4f}")