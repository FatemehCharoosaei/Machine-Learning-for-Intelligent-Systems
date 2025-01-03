# -*- coding: utf-8 -*-


import pandas as pd
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import r2_score, mean_absolute_error
import numpy as np 

scale = StandardScaler()

df = pd.read_excel('cars.xls')
df = df.sample(frac=1)


df['Model_ord'] = pd.Categorical(df.Model).codes
df['Make_ord'] = pd.Categorical(df.Make).codes
df['Trim_ord'] = pd.Categorical(df.Trim).codes
df['Type_ord'] = pd.Categorical(df.Type).codes

X = df[['Mileage', 'Cylinder', 'Liter', 'Doors', 'Cruise', 'Model_ord', 'Make_ord', 'Trim_ord', 'Type_ord', 'Sound', 'Leather']]
y = df['Price']

c = 120
X_train = X[:-c]
y_train = y[:-c]

X_predict = X[-c:]
y_predict = y[-c:]

#X_train[['Mileage', 'Cylinder', 'Doors', 'Model_ord']] = scale.fit_transform(X_train[['Mileage', 'Cylinder', 'Doors', 'Model_ord']].as_matrix())

print (X_train)

model = sm.OLS(y_train, X_train)
est = model.fit()

#X_predict[['Mileage', 'Cylinder', 'Doors', 'Model_ord']] = scale.transform(X_predict[['Mileage', 'Cylinder', 'Doors', 'Model_ord']].as_matrix())

predicted = est.predict(X_predict)
est.summary()

#y.groupby(df.Doors).mean()
r_s = r2_score(y_predict, predicted)
plt.scatter(predicted, y_predict)

MAE = mean_absolute_error(y_predict, predicted)

# Calculate mean absolute percentage error (MAPE)
errors = abs(predicted - y_predict)
mape = 100 * (errors / y_predict)

# Calculate and display accuracy
accuracy = 100 - np.mean(mape)
print('Accuracy:', round(accuracy, 2), '%.')
