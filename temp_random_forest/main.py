# -*- coding: utf-8 -*-


import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error


features_df = pd.read_csv('temps.csv')

features_df = pd.get_dummies(features_df)

labels = np.array(features_df['actual'])
features_df = features_df.drop('actual', axis = 1)
feature_list = list(features_df.columns)
features = np.array(features_df)

train_features, test_features, train_labels, test_labels = train_test_split(features, labels, test_size = 0.25)


rf_model = RandomForestRegressor(n_estimators = 1000)
rf_model.fit(train_features, train_labels)
predictions = rf_model.predict(test_features)


r_s = r2_score(test_labels, predictions)

# Calculate mean absolute percentage error (MAPE)
errors = abs(predictions - test_labels)
mape = 100 * (errors / test_labels)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')

MAE = mean_absolute_error(test_labels, predictions)





