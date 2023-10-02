import BoW as B
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier


def give_labels(file):
    '''
    labels the given data as the given file name
    '''
    df = pd.read_fwf(file, index=False)
    df['label'] = file
    return df

def Extract(lst):
    '''
    Helper function to get all index 0 from sublist
    '''
    return [item[0] for item in lst]

#Combines training data subjets. One complete text file can be later done in its own script
datasets = ["optimistic_crab.txt", "programmer.txt", "sad_keanu.txt", "stare_dad.txt"]
panda_datasets = []
for f in datasets:
    panda_datasets.append(give_labels(f))

result = pd.concat(panda_datasets)
result = result.drop(["Unnamed: 1","Unnamed: 2","Unnamed: 3","Unnamed: 4","Unnamed: 5","Unnamed: 6","Unnamed: 7","Unnamed: 8","Unnamed: 9","Unnamed: 10","Unnamed: 11","Unnamed: 12"], axis=1)
result[['Text', 'Labels']] = result[['Text', 'label']].astype(str)
Bag = B.BoW()
tr = Bag.fit_transform(Extract(result.values.tolist()))

#Split the training data
X = tr
y = result['label'].to_numpy()
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#Train the model
mlp = MLPClassifier(hidden_layer_sizes=(4426, 1106), activation='relu', solver='adam', max_iter=1000).fit(X_train, y_train)
score = mlp.score(X_test, y_test)
print(score)