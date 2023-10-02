import BoW as B
import pandas as pd

Data = []
with open('creepy_wonka.txt') as f:
    Data.append(f.readlines())

Bag = B.BoW()
tr = Bag.fit_transform(Data[0])

with open('hello.txt', 'w') as f:
    for line in tr:
        f.write(str(len(line)))
        f.write(" ")
        f.write(str(sum(line)))
        f.write(" ")
        f.write(str(max(line)))
        f.write('\n')