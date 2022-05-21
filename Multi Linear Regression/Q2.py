import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn import preprocessing
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error

Startups = pd.read_csv("50_startups.csv")

X = Startups.iloc[0:,0:4]
Y = Startups["Profit"]

label_encoder = preprocessing.LabelEncoder()

X["State"] = label_encoder.fit_transform(X["State"])

X_train, X_test, Y_train, Y_test= train_test_split(X, Y, test_size= 0.5, random_state=0) 

regressor = LinearRegression()
regressor.fit(X_train, Y_train)

Y_pred = regressor.predict(X_test)

print('Train Score: ', regressor.score(X_train, Y_train))
print('Test Score: ', regressor.score(X_test, Y_test))

RC = regressor.coef_

print("Y = {}*X1+{}*X2+{}*X3+{}*X4".format(RC[0],RC[1],RC[2],RC[3]))

score = r2_score(Y_test,Y_pred)

print("R^2 Score : ",score)
print("Mean Squared Error : ",mean_squared_error(Y_test,Y_pred))
print("Root Mean Squared Error : ",np.sqrt(mean_squared_error(Y_test,Y_pred)))
