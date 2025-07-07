from pathlib import Path
import pandas as pd
from sklearn.decomposition import PCA
import plotly.express as px
from scipy.io import arff

base_dir = Path('../datasets')
groups = []

for orig_path in base_dir.glob('*/**/*original*.arff'):
    folder = orig_path.parent
    prefix = orig_path.stem.replace('_original', '')
    noisy_files = sorted(folder.glob(f'{prefix}_*.csv'))
    groups.append((orig_path, noisy_files))

print(f"Found {len(groups)} dataset groups.")

for i, (orig_path, noisy_paths) in enumerate(groups):
    print(f"\nShowing group {i + 1}/{len(groups)}: {orig_path.name}")
    
    # Load original
    data, _ = arff.loadarff(orig_path)
    df_orig = pd.DataFrame(data)
    for col in df_orig.select_dtypes(include='object').columns:
        df_orig[col] = df_orig[col].str.decode('utf-8')

    target_col = df_orig.columns[-1]
    X_orig = df_orig.drop(columns=target_col)
    y_orig = df_orig[target_col]

    pca = PCA(n_components=2)
    X_orig_pca = pca.fit_transform(X_orig)
    fig = px.scatter(x=X_orig_pca[:, 0], y=X_orig_pca[:, 1], color=y_orig.astype(str),
                     title=f'Original: {orig_path.name}')
    fig.show()

    for path in noisy_paths:
        df = pd.read_csv(path)
        X = df.drop(columns=target_col, errors='ignore')
        y = df[target_col] if target_col in df.columns else 'unknown'
        X_pca = pca.transform(X)
        fig = px.scatter(x=X_pca[:, 0], y=X_pca[:, 1], color=y.astype(str),
                         title=f'Noisy: {path.name}')
        fig.show()

    input("Press ENTER to show next dataset group...")
