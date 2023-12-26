# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 11:43:56 2023

@author: FAUZIAH
"""

import cx_Oracle
%matplotlib inline
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

conn = cx_Oracle.connect('xhq/xhq2mes@akuas952.office.graphiteelectrodes.net/mes301')
cur = conn.cursor()

sql_st = """
SELECT RECV_GREEN_LENGTH, PI_MEASLENGTH_MM, FIN_LENGTH_IN_MM, RECV_GREEN_DIAMETER_H AS GREEN_DIAMETER, 
(RECV_GREEN_LENGTH-PI_MEASLENGTH_MM) AS BAKESHRINKAGE,
(PI_MEASLENGTH_MM-FIN_LENGTH_IN_MM) AS GRAPHSHRINKAGE
FROM vw_mes_total
WHERE
(RECV_GREEN_LENGTH-PI_MEASLENGTH_MM)>0 AND (RECV_GREEN_LENGTH-PI_MEASLENGTH_MM)<80  AND 
(PI_MEASLENGTH_MM-FIN_LENGTH_IN_MM)>0 AND (PI_MEASLENGTH_MM-FIN_LENGTH_IN_MM)<80  AND 
RECV_GREEN_LENGTH>0 AND PI_MEASLENGTH_MM>0 AND FIN_LENGTH_IN_MM>0 AND RECV_GREEN_DIAMETER_H >0 AND
FIN_MACHINED_ENDDATE > '01-01-2020'
ORDER BY FIN_MACHINED_ENDDATE DESC """


df = pd.read_sql_query(sql_st, con=conn)
print(df)
print(df.columns)

# Split the data into features and target variable
X = df[['PI_MEASLENGTH_MM', 'FIN_LENGTH_IN_MM', 'GREEN_DIAMETER']]
y = df['RECV_GREEN_LENGTH']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
linear_model = LinearRegression()
random_forest_model = RandomForestRegressor()

# Train the models
linear_model.fit(X_train, y_train)
random_forest_model.fit(X_train, y_train)

# Make predictions
linear_predictions = linear_model.predict(X_test)
random_forest_predictions = random_forest_model.predict(X_test)

# Evaluate performance
linear_mse = mean_squared_error(y_test, linear_predictions)
random_forest_mse = mean_squared_error(y_test, random_forest_predictions)

linear_r2 = r2_score(y_test, linear_predictions)
random_forest_r2 = r2_score(y_test, random_forest_predictions)

print("Linear Regression MSE:", linear_mse)
print("Random Forest MSE:", random_forest_mse)

print("Linear Regression R^2:", linear_r2)
print("Random Forest R^2:", random_forest_r2)

# Create a DataFrame to compare the results
results = pd.DataFrame({'Model': ['Linear Regression', 'Random Forest', 'Ridge Regression'],
                        'MSE': [linear_mse, random_forest_mse, mse ],
                        'RMSE':[linear_rmse, random_forest_rmse, ridge_regression_rmse ],
                        'R^2': [linear_r2, random_forest_r2, r2]})

print(results)

random_forest_model.predict([[3031, 2996, 741]])
linear_model.predict([[3031, 2996, 741]])

## ADDING RIDGE REGRESSION
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error

# Initialize the Ridge regression model
ridge_model = Ridge(alpha=1.0) 

# Train the model
ridge_model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = ridge_model.predict(X_test)

# Calculate mean squared error (MSE)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
print(rmse)

print("Ridge Regression MSE:", mse)

# Calculate R^2 score
r2 = r2_score(y_test, y_pred)

print("R^2 score:", r2)

#RMSE ALL MODEL
linear_rmse = np.sqrt(linear_mse)
random_forest_rmse = np.sqrt(random_forest_mse)
ridge_regression_rmse = np.sqrt(mse)






