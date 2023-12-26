# -*- coding: utf-8 -*-
"""
Created on Thu Jul  6 10:03:31 2023

@author: FAUZIAH
"""

import cx_Oracle
%matplotlib inline
import pandas as pd
import numpy as np
import itertools
import statsmodels.api as sm
import matplotlib.pyplot as plt

#GET THE DATA 
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

X = df[[ 'PI_MEASLENGTH_MM', 'FIN_LENGTH_IN_MM','GREEN_DIAMETER', 'BAKESHRINKAGE', 'GRAPHSHRINKAGE']]
X = X.astype(float)
y = pd.DataFrame(df.RECV_GREEN_LENGTH)
y = y.astype(float)
 #Model selection using the Validation Set Approach
np.random.seed(seed=12)
train = np.random.choice([True, False], size = len(y), replace = True)
test = np.invert(train)

def processSubset(feature_set, X_train, y_train, X_test, y_test):
    # Fit model on feature_set and calculate RSS
    model = sm.OLS(y_train,X_train[list(feature_set)])
    regr = model.fit()
    RSS = ((regr.predict(X_test[list(feature_set)]) - y_test) ** 2).sum()
    return {"model":regr, "RSS":RSS}

def forward(predictors, X_train, y_train, X_test, y_test):
    
    results = []

    # Pull out predictors we still need to process
    remaining_predictors = [p for p in X_train.columns if p not in predictors]
    
    for p in remaining_predictors:
        results.append(processSubset(predictors+[p], X_train, y_train, X_test, y_test))
    
    # Wrap everything up in a nice dataframe
    models = pd.DataFrame(results)
    
    # Choose the model with the highest RSS
    best_model = models.loc[models['RSS'].argmin()]
        
    # Return the best model, along with some other useful information about the model
    return best_model



models_train = pd.DataFrame(columns=["RSS", "model"])
predictors = []
for i in range(1,len(X.columns)+1):    
    models_train.loc[i] = forward(predictors, X[train], y[train]["RECV_GREEN_LENGTH"], X[test], y[test]["RECV_GREEN_LENGTH"])
    predictors = models_train.loc[i]["model"].model.exog_names
    
print(predictors)
print(models_train)


plt.plot(models_train["RSS"])
plt.xlabel('# Predictors')
plt.ylabel('RSS')
plt.plot(models_train["RSS"].astype(float).argmin(), models_train["RSS"].min(), "or")







