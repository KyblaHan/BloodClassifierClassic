from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample

import matplotlib.pyplot as plt
import pandas as pd
from classifier import load_and_preprocess_data

X, y = load_and_preprocess_data("ProjectData/Data/train_multiclass.csv", 1)

clf = RandomForestClassifier()
clf.fit(X, y)
params = clf.feature_importances_
x = 0
for p in params:
    x += p
x = x / 702
print(x)

best = []
for i in range(0,702):
    if params[i]>=x:
        best.append(i)
print(len(X[1]))
X = X[:, best]
print(len(X[1]))
