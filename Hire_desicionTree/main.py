# -*- coding: utf-8 -*-


import numpy as np 
import pandas as pd
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier


df = pd.read_csv('PastHires.csv')

d = {'Y': 1, 'N': 0}
df['Hired'] = df['Hired'].map(d)
df['Employed?'] = df['Employed?'].map(d)
df['Top-tier school'] = df['Top-tier school'].map(d)
df['Interned'] = df['Interned'].map(d)
d = {'BS': 0, 'MS': 1, 'PhD': 2}
df['Level of Education'] = df['Level of Education'].map(d)
features = list(df.columns[:6])
y = df["Hired"]
X = df[features]

#### Decision Tree Classification
tree_model = tree.DecisionTreeClassifier()
tree_model.fit(X,y)


X_test = np.array([[1, 0, 2, 1, 1, 0]])
tree_predictions = tree_model.predict(X_test)


#### Random Forest Classification

rForest_model = RandomForestClassifier(n_estimators=10)
rForest_model.fit(X, y)
rForest_predictions = rForest_model.predict(X_test)

