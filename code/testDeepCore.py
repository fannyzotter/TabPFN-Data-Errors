import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import LabelEncoder
from deepcore.methods import SubmodularSelection
import torch.nn as nn

# 1. CSV-Daten laden
df = pd.read_csv("archive/XAI_Drilling_Dataset.csv")

# features are in the first line 
features = df.columns[:-1]  # Alle Spalten außer der letzten
target = df.columns[-1]     # Letzte Spalte als Zielvariable


X = df[features].values
y = LabelEncoder().fit_transform(df[target].values)

# 3. Dataset definieren
class TabularDataset(Dataset):
    def __init__(self, features, labels):
        self.X = torch.tensor(features, dtype=torch.float32)
        self.y = torch.tensor(labels, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

dataset = TabularDataset(X, y)

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

model = SimpleMLP(input_dim=X.shape[1], num_classes=len(set(y)))

# 5. Coreset-Auswahl mit DeepCore
selector = SubmodularSelection(model=model, dataset=dataset, fraction=0.1, method='GraphCut')
selected_indices = selector.select()

# 6. Coreset extrahieren
coreset = torch.utils.data.Subset(dataset, selected_indices)