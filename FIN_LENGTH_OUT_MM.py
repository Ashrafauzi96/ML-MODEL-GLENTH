import cx_Oracle
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error
import pickle

# import data
df = pd.read_csv('C:/Users/fauziah/Desktop/export2.csv')

# split data into x and y variable
X = df[['RECV_GREEN_LENGTH', 'PI_MEASLENGTH_MM', 'FIN_LENGTH_IN_MM', 'GREEN_DIAMETER']]
y = df['FIN_LENGTH_OUT_MM']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
random_forest_model = RandomForestRegressor()

# Train the models
random_forest_model.fit(X_train, y_train)

# Make pickle file of our model
pickle.dump(random_forest_model, open("model_test.pkl", "wb"))



