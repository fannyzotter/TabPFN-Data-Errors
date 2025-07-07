import pandas as pd
from scipy.io import arff
from badgers.generators.tabular_data.noise import GaussianNoiseGenerator
from badgers.generators.tabular_data.missingness import MissingCompletelyAtRandom, DummyMissingNotAtRandom, DummyMissingAtRandom
from sklearn.preprocessing import LabelEncoder
from badgers.generators.tabular_data.outliers import *
from numpy.random import default_rng
from pathlib import Path

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

def MyOutlierGenerator(X, y, number_outliers, target_col, output_dir, prefix):
    for num in number_outliers:
        transformer = ZScoreSamplingGenerator(random_generator=default_rng(42))
        Xo, _ = transformer.generate(X.copy(), y=y, n_outliers=num)
        df_outliers = pd.DataFrame(Xo)
        df_outliers[target_col] = y
        filename = output_dir / f"{prefix}_Outlier_std{int(num)}.csv"
        df_outliers.to_csv(filename, index=False)

if __name__ == "__main__":

    NOISE_LEVELS = [0.01, 0.05, 0.1]
    MISSING_VALUES = [0.01, 0.05, 0.1]
    OUTLIER_NUMBERS = [10, 50, 100]

    base_dir = Path('../datasets')
    for arff_path in base_dir.glob('*/**/*original*.arff'):
        data, meta = arff.loadarff(arff_path)
        df = pd.DataFrame(data)
    
        for col in df.select_dtypes(include=['object']).columns:
            df[col] = df[col].str.decode('utf-8')
    
        output_dir = arff_path.parent
        prefix = arff_path.stem.replace('_original', '')

        # Try to infer target column (last column is common)
        target_col = df.columns[-1]
        X = df.drop(columns=target_col)
        y = df[target_col]

        # Apply all noise types
        MyGaussianNoiseGenerator(X, y, NOISE_LEVELS, target_col, output_dir, prefix)
        MyMissingCompletelyAtRandom(X, y, MISSING_VALUES, target_col, output_dir, prefix)
        MyOutlierGenerator(X, y, OUTLIER_NUMBERS, target_col, output_dir, prefix)