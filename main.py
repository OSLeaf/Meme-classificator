import BoW as B
import pandas as pd
import numpy as np


def give_labels(file):
    df = pd.read_fwf(file, index=False)
    df['label'] = file
    return df

datasets = ["optimistic_crab.txt", "programmer.txt", "sad_keanu.txt", "stare_dad.txt"]
panda_datasets = []
for f in datasets:
    panda_datasets.append(give_labels(f))

result = pd.concat(panda_datasets)
result = result.drop(["Unnamed: 1","Unnamed: 2","Unnamed: 3","Unnamed: 4","Unnamed: 5","Unnamed: 6","Unnamed: 7","Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12"], axis=1)
print(result.sample(n=5))
result[['Text', 'Labels']] = result[['Text', 'label']].astype(str)
Bag = B.BoW()
tr = Bag.fit_transform(result.values.tolist()[0])

X = tr
y = result.drop(['Text'], axis=1).to_numpy()

