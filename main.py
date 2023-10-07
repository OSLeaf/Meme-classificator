import BoW as B
import pandas as pd
import numpy as np
import functools
import json
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
datasets = ["optimistic_crab.txt", "programmer.txt", "sad_keanu.txt", "stare_dad.txt", "creepy_wonka.txt"]
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
mlp = MLPClassifier(hidden_layer_sizes=(20, 20), activation='relu', solver='adam', max_iter=1000).fit(X_train, y_train)
score = mlp.score(X_test, y_test)
print(score)
master = [i.tolist() for i in mlp.coefs_]
json_array = json.dumps(master)

with open("coefs.json", "w") as outfile:
    outfile.write(json_array)

f = open("labels.txt", "w")
f.write(str(mlp.classes_))

tests = ["Oh, You just graduated? You must know everything.", "TAt least when I don't come home tonight, my wife and kids won't know how horrible I died...  Is that a camera? Fuck", "Ok, so the code didn't run properly. I'll add a print statement to check what it is doing It runs perfectly", "I hooked up with this girl i met online, she said she had a body built for sin. Too bad that the sin was gluttony" ,"Dad, i was fingering mum and i found your wedding ring That's not right ? ... I'm sure i left that inside of your little brother."]
for test in tests:
    test = Bag.transform(test)
    resultp = mlp.predict(test.reshape(1, -1))
    print(resultp)
    resultc = functools.reduce(lambda a, b: np.dot(a, b), mlp.coefs_, test)
    print(resultc)
