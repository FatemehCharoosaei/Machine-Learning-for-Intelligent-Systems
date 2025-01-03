# -*- coding: utf-8 -*-


import numpy as np 
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans 
scores = np.loadtxt('students_scores.txt', delimiter=',')
scores = scores[:,0:2]
plt.scatter(scores[:, 0], scores[:, 1], s=50, color=(.5, .2, .8) ,alpha=.5)

model = KMeans(n_clusters=3)
model = model.fit(scores)

plt.scatter(scores[:, 0], scores[:, 1], c=model.labels_.astype(np.float))
