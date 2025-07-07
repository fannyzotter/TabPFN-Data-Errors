import arff
import pandas as pd

path = '/home/zotter/fanny/TabPFN-Data-Errors/datasets/'
name = 'aps_failure'
version = '_v1'

# .arff-Datei laden
with open(path + name + '.arff') as f:
    dataset = arff.load(f)

# In DataFrame umwandeln
df = pd.DataFrame(dataset['data'], columns=[attr[0] for attr in dataset['attributes']])

# Als .csv speichern
df.to_csv(path + name + version + '_original.csv', index=False)
