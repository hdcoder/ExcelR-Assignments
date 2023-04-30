import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import keras
import tensorflow as tf
import warnings

from keras.wrappers.scikit_learn import KerasRegressor
from keras.models import Sequential
from keras.layers import InputLayer,Dense

from sklearn.model_selection import cross_val_score, KFold, train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import StandardScaler

def baseline_model():
    model = Sequential()
    model.add(Dense(10, input_dim=10, activation='tanh'))
    model.add(Dense(1))
    model.compile(loss='mean_squared_error', optimizer='adam')
    return model

df = pd.read_csv("Gasturbines.csv")
print(df.head())

print(df.dtypes)

print(df.columns)

warnings.filterwarnings("ignore")
fig, axes = plt.subplots(5, 2, figsize=(20, 15))
fig.suptitle('Univariate Analysis',fontsize=20)

sns.distplot(df['AT'],ax=axes[0,0],color='indigo')
sns.distplot(df['AP'],ax=axes[0,1],color='orange')
sns.distplot(df['AH'],ax=axes[1,0],color='indigo')
sns.distplot(df['AFDP'],ax=axes[1,1],color='orange')
sns.distplot(df['GTEP'],ax=axes[2,0],color='indigo')
sns.distplot(df['TIT'],ax=axes[2,1],color='orange')
sns.distplot(df['TAT'],ax=axes[3,0],color='indigo')
sns.distplot(df['CDP'],ax=axes[3,1],color='orange')
sns.distplot(df['CO'],ax=axes[4,0],color='indigo')
sns.distplot(df['NOX'],ax=axes[4,1],color='orange')

sns.distplot(df['TEY'],color='orange')

fig, axes = plt.subplots(5, 2, figsize=(20, 20))
fig.suptitle('Bivariate Analysis',fontsize=20)

sns.regplot(x="AT",y="TEY",data=df,ax=axes[0,0],color='indigo')
sns.regplot(x="AP",y="TEY",data=df,ax=axes[0,1],color='orange')
sns.regplot(x="AH",y="TEY",data=df,ax=axes[1,0],color='indigo')
sns.regplot(x="AFDP",y="TEY",data=df,ax=axes[1,1],color='orange')
sns.regplot(x="GTEP",y="TEY",data=df,ax=axes[2,0],color='indigo')
sns.regplot(x="TIT",y="TEY",data=df,ax=axes[2,1],color='orange')
sns.regplot(x="TAT",y="TEY",data=df,ax=axes[3,0],color='indigo')
sns.regplot(x="CDP",y="TEY",data=df,ax=axes[3,1],color='orange')
sns.regplot(x="CO",y="TEY",data=df,ax=axes[4,0],color='indigo')
sns.regplot(x="NOX",y="TEY",data=df,ax=axes[4,1],color='orange')

fig, axes = plt.subplots(figsize=(10, 8))
sns.heatmap(df.corr(),annot=True)

X = df.loc[:,['AT', 'AP', 'AH', 'AFDP', 'GTEP', 'TIT', 'TAT', 'CDP', 'CO','NOX']]
y= df.loc[:,['TEY']]

scaler = StandardScaler()
X = scaler.fit_transform(X)
y = scaler.fit_transform(y)

estimator = KerasRegressor(build_fn=baseline_model, nb_epoch=50, batch_size=100, verbose=False)
kfold = KFold(n_splits=10)
results = cross_val_score(estimator, X, y, cv=kfold)
print("Results: %.2f (%.2f) MSE" % (results.mean(), results.std()))

estimator.fit(X, y)
prediction = estimator.predict(X)

print(prediction)