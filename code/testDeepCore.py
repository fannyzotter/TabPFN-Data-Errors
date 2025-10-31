import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
#from external.deepcore.deepcore.methods import k_center_greedy
import torch.nn as nn
import os

# 3. Dataset definieren
class TabularDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# 4. Einfaches MLP-Modell definieren
class SimpleMLP(nn.Module):
    def __init__(self, input_dim, num_classes):
        super(SimpleMLP, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.fc(x)

def run_deepcore_subset(dataset_name, base_path, fraction=0.5):
    input_path = base_path / "datasets" / dataset_name
    output_path = base_path / "kCenterSubsets" / dataset_name
    
    # Erstelle Output-Verzeichnis
    output_path.mkdir(parents=True, exist_ok=True)

    # Iteriere über alle CSV-Dateien im Dataset-Ordner
    for filename in os.listdir(input_path):
        if not filename.endswith(".csv"):
            print('no file under this filename')
            continue

        file_path = input_path / filename
        out_file = output_path / filename.replace(".csv", f"_kcenter.csv")

        try:
            df = pd.read_csv(file_path)

            features = df.columns[:-1]
            target = df.columns[-1]

            X = df[features].values
            y = LabelEncoder().fit_transform(df[target].values)

            dataset = TabularDataset(X, y)
            model = SimpleMLP(input_dim=X.shape[1], num_classes=len(set(y)))

#            selector = k_center_greedy(model=model, dataset=dataset, fraction=fraction)
#            selected_indices = selector.select()
#            print(f"Selected {len(selected_indices)} samples using DeepCore.")
#            #save coreset to csv
#            coreset_df = pd.DataFrame(X[selected_indices], columns=features)
#            coreset_df[target] = y[selected_indices]
#
#            coreset_df.to_csv(out_file, index=False)
#            print(f"DeepCore subset saved to {out_file}")
        except Exception as e:
            print(f"Fehler beim Verarbeiten von {filename}: {e}")
            continue