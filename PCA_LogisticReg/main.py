# -*- coding: utf-8 -*-

import os
from time import time
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

mnist_path = 'Fashion_mnist/'

train_file = 'fashion-mnist_train.csv'
train_pd = pd.read_csv(os.path.join(mnist_path, train_file))
train_data = train_pd.loc[:,'pixel1':'pixel784'].values
train_label = train_pd.loc[:,'label'].values

mask = np.logical_or(train_label == 2, train_label == 1)

train_data = train_data[mask]
train_label = train_label[mask]

test_file = 'fashion-mnist_test.csv'
test_pd = pd.read_csv(os.path.join(mnist_path, test_file))
test_data = test_pd.loc[:,'pixel1':'pixel784'].values
test_label = test_pd.loc[:,'label'].values

mask = np.logical_or(test_label == 2, test_label == 1)
test_data = test_data[mask]
test_label = test_label[mask]

sample = train_data[1, :]
sample = sample.reshape(28, 28)
plt.imshow(sample)
plt.show

train_data_norm = train_data/255.0
test_data_norm = test_data/255.0


######## Logistic Regressoin Classification ##########

t0=time()
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_data, train_label)

predications = logisticRegr.predict(test_data)
acc1 = accuracy_score(predications, test_label) * 100
d0 = round(time()-t0, 3)


######## PCA: Principle Component Analysis ###########

#pca = PCA(n_components=2)
#pca.fit(train_data)
#train_features = pca.transform(train_data)
#test_features = pca.transform(test_data)
#
#train_features_df = pd.DataFrame(data=train_features, columns=['PC1', 'PC2'])
#train_label_df = pd.DataFrame(data=train_label, columns=['label'])
#final_df = pd.concat([train_features_df, train_label_df], axis=1)
#
#
#fig = plt.figure(figsize = (8,8))
#ax = fig.add_subplot(1,1,1) 
#ax.set_xlabel('Principal Component 1', fontsize = 15)
#ax.set_ylabel('Principal Component 2', fontsize = 15)
#ax.set_title('2 Component PCA', fontsize = 20)
#
#
#for label in range(10):
#    indicesToKeep = final_df['label'] == label
#    ax.scatter(final_df.loc[indicesToKeep, 'PC1']
#               , final_df.loc[indicesToKeep, 'PC2']
#               , s = 5
#               , c = [(1/(label+1),1-(1/(label+1)) , (1/(label+1))),]
#               )
#ax.grid()
#plt.show()



######## Logistic Reg Classification and PCA ##########

pca = PCA(n_components=20)
pca.fit(train_data)
train_features = pca.transform(train_data)


t1=time()
test_features = pca.transform(test_data)
logisticRegr = LogisticRegression(solver = 'lbfgs')
logisticRegr.fit(train_features, train_label)

predications = logisticRegr.predict(test_features)
acc2 = accuracy_score(predications, test_label) * 100
d1 = round(time()-t1, 3)
print (d0/d1)




