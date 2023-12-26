import pandas as pd
import openpyxl
import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import pickle
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np

#load the csv file
df = pd.read_excel("green_length.xlsx")

#select independent and dependent variable
X = df[['PI_MEASLENGTH_MM', 'FIN_LENGTH_IN_MM', 'GREEN_DIAMETER']]
y = df['RECV_GREEN_LENGTH']

from sklearn.model_selection import train_test_split
X_train, x_test, Y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the models
random_forest_model = RandomForestRegressor()

# Train the models
random_forest_model.fit(X_train.values, Y_train)

# RMSE Value
"""""
random_forest_predictions = random_forest_model.predict(x_test)

random_forest_mse = mean_squared_error(y_test, random_forest_predictions)

random_forest_r2 = r2_score(y_test, random_forest_predictions)

random_forest_rmse = np.sqrt(random_forest_mse)

print(random_forest_rmse)
"""""

# Make pickle file of our model
pickle.dump(random_forest_model, open("model.pkl", "wb"))
#random_forest_predictions = random_forest_model.predict(x_test)

#print(random_forest_model.predict([[2592, 2527, 641]]))
