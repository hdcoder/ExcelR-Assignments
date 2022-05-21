import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

Corolla = pd.read_csv("ToyotaCorolla.csv")

X = Corolla[["Age_08_04","KM","HP","cc","Doors","Gears","Quarterly_Tax","Weight"]]
Y = Corolla["Price"]

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.5, random_state=0) 

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

RC = regressor.coef_

print("Y = {}*X1+{}*X2+{}*X3+{}*X4+{}*X5+{}*X6+{}*X7+{}*X8".format(RC[0],RC[1],RC[2],RC[3],RC[4],RC[5],RC[6],RC[7]))

score = r2_score(Y_test,Y_pred)

print("R^2 Score : ",score)
print("Mean Squared Error : ",mean_squared_error(Y_test,Y_pred))
print("Root Mean Squared Error : ",np.sqrt(mean_squared_error(Y_test,Y_pred)))