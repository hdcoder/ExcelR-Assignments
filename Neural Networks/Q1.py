import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import GridSearchCV, KFold

from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasClassifier
from keras.layers import Dropout

from tensorflow.keras.optimizers import Adam

import seaborn as sns

def create_model():

    Model = Sequential()
    Model.add(Dense(12, input_dim=28, activation='relu'))
    Model.add(Dense(8, activation='relu'))
    Model.add(Dense(1, activation='sigmoid'))
    Adam1 = Adam(lr=0.01)
    Model.compile(loss='binary_crossentropy', optimizer=Adam1, metrics=['accuracy'])
    return Model

def create_model(learning_rate,dropout_rate):

    Model = Sequential()
    Model.add(Dense(8,input_dim = 28,kernel_initializer = 'normal',activation = 'relu'))
    Model.add(Dropout(dropout_rate))
    Model.add(Dense(4,input_dim = 28,kernel_initializer = 'normal',activation = 'relu'))
    Model.add(Dropout(dropout_rate))
    Model.add(Dense(1,activation = 'sigmoid'))
    Adam1 = Adam(lr = learning_rate)
    Model.compile(loss = 'binary_crossentropy',optimizer = Adam1,metrics = ['accuracy'])
    return Model

def create_model(activation_function,init):

    Model = Sequential()
    Model.add(Dense(8,input_dim = 28,kernel_initializer = init,activation = activation_function))
    Model.add(Dropout(0.1))
    Model.add(Dense(4,input_dim = 28,kernel_initializer = init,activation = activation_function))
    Model.add(Dropout(0.1))
    Model.add(Dense(1,activation = 'sigmoid'))
    Adam1 = Adam(lr = 0.001)
    Model.compile(loss = 'binary_crossentropy',optimizer = Adam1,metrics = ['accuracy'])
    return Model

def create_model(neuron1,neuron2):

    Model = Sequential()
    Model.add(Dense(neuron1,input_dim = 28,kernel_initializer = 'uniform',activation = 'tanh'))
    Model.add(Dropout(0.1))
    Model.add(Dense(neuron2,input_dim = neuron1,kernel_initializer = 'uniform',activation = 'tanh'))
    Model.add(Dropout(0.1))
    Model.add(Dense(1,activation = 'sigmoid'))
    Adam1 = Adam(lr = 0.001)
    Model.compile(loss = 'binary_crossentropy',optimizer = Adam1,metrics = ['accuracy'])
    return Model

def create_model():

    Model = Sequential()
    Model.add(Dense(16,input_dim = 28,kernel_initializer = 'uniform',activation = 'tanh'))
    Model.add(Dropout(0.1))
    Model.add(Dense(4,input_dim = 16,kernel_initializer = 'uniform',activation = 'tanh'))
    Model.add(Dropout(0.1))
    Model.add(Dense(1,activation = 'sigmoid'))
    Adam1 = Adam(lr = 0.001)
    Model.compile(loss = 'binary_crossentropy',optimizer = Adam1,metrics = ['accuracy'])
    return Model

def create_model(learning_rate,dropout_rate,activation_function,init,neuron1,neuron2):
    Model = Sequential()
    Model.add(Dense(neuron1,input_dim = 28,kernel_initializer = init,activation = activation_function))
    Model.add(Dropout(dropout_rate))
    Model.add(Dense(neuron2,input_dim = neuron1,kernel_initializer = init,activation = activation_function))
    Model.add(Dropout(dropout_rate))
    Model.add(Dense(1,activation = 'sigmoid'))
    Adam1 = Adam(lr = learning_rate)
    Model.compile(loss = 'binary_crossentropy',optimizer = Adam1,metrics = ['accuracy'])
    return Model

FF_data = pd.read_csv("Forestfires.csv", delimiter=",")
print(FF_data.head())
print()

print(FF_data.info())
print()

print(FF_data.describe())
print()

plt.figure(figsize=(20,10))
sns.heatmap(FF_data.corr(),annot=True)

print(FF_data.shape)
print()

FF_df = FF_data.copy()
print(FF_df.head(2))
print()

FF_df = FF_df.drop(columns=['month','day'], axis=1)
print(FF_df.head(2))
print()

print(FF_df.columns)
print()

print(FF_df.size_category.value_counts())
print()

print(FF_df.area.value_counts())
print()

print(FF_df.rain.value_counts())
print()

Label_encoder = preprocessing.LabelEncoder()
FF_df['size_category']= Label_encoder.fit_transform(FF_df['size_category'])
print(FF_data.info())
print()

X = FF_df.drop('size_category', axis=1)
Y = FF_df['size_category']
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.20)

Model = Sequential()
Model.add(Dense(42, input_shape=(28,), activation = 'relu'))
Model.add(Dense(28, activation='relu'))
Model.add(Dense(1, activation='sigmoid'))
Model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
History = Model.fit(X_train,Y_train, validation_split=0.33, epochs=180, batch_size=10)

Scores = Model.evaluate(X_train, Y_train)
print("%s: %.2f%%" % (Model.metrics_names[1], Scores[1]*100))
print()

Scores = Model.evaluate(X_train, Y_train)
print("%s: %.2f%%" % (Model.metrics_names[1], Scores[1]*100))
print()

Scores = Model.evaluate(X_test, Y_test)
print("%s: %.2f%%" % (Model.metrics_names[1], Scores[1]*100))
print()

print(History.history.keys())
print()

Model.compile(loss="categorical_crossentropy",optimizer='rmsprop', metrics=["accuracy"])

plt.figure()
plt.plot(History.history['accuracy'])
plt.plot(History.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.figure()
plt.plot(History.history['loss'])
plt.plot(History.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

X1 = FF_df.drop('size_category', axis=1)
Y1 = FF_df['size_category']
A = StandardScaler()
A.fit(X1)
X_standardized = A.transform(X1)
print(pd.DataFrame(X_standardized).describe())

Model = KerasClassifier(build_fn = create_model,verbose = 0)
Batch_size = [10,20,40]
Epochs = [10,50,100]
Param_grid = dict(batch_size = Batch_size,epochs = Epochs)
Grid = GridSearchCV(estimator = Model,param_grid = Param_grid,cv = KFold(),verbose = 10)
Grid_result = Grid.fit(X_standardized,Y1)
print('Best : {}, using {}'.format(Grid_result.best_score_,Grid_result.best_params_))
print()

Means = Grid_result.cv_results_['mean_test_score']
Stds = Grid_result.cv_results_['std_test_score']
Params = Grid_result.cv_results_['params']

for Mean, Stdev, Param in zip(Means, Stds, Params):
    print('{},{} with: {}'.format(Mean, Stdev, Param))
print()

Model = KerasClassifier(build_fn = create_model,verbose = 0,batch_size = 40,epochs = 10)
Learning_rate = [0.001,0.01,0.1]
Dropout_rate = [0.0,0.1,0.2]
Param_grids = dict(learning_rate = Learning_rate,dropout_rate = Dropout_rate)
Grid = GridSearchCV(estimator = Model,param_grid = Param_grids,cv = KFold(),verbose = 10)
Grid_result = Grid.fit(X_standardized,Y1)
print('Best : {}, using {}'.format(Grid_result.best_score_,Grid_result.best_params_))
print()

Means = Grid_result.cv_results_['mean_test_score']
Stds = Grid_result.cv_results_['std_test_score']
Params = Grid_result.cv_results_['params']

for mean, stdev, param in zip(Means, Stds, Params):
  print('{},{} with: {}'.format(mean, stdev, param))
print()

Model = KerasClassifier(build_fn = create_model,verbose = 0,batch_size = 40,epochs = 10)
Activation_function = ['softmax','relu','tanh','linear']
Init = ['uniform','normal','zero']
Param_grids = dict(activation_function = Activation_function,init = Init)
Grid = GridSearchCV(estimator = Model,param_grid = Param_grids,cv = KFold(),verbose = 10)
Grid_result = Grid.fit(X_standardized,Y)
print('Best : {}, using {}'.format(Grid_result.best_score_,Grid_result.best_params_))
print()

Means = Grid_result.cv_results_['mean_test_score']
Stds = Grid_result.cv_results_['std_test_score']
Params = Grid_result.cv_results_['params']

for Mean, Stdev, Param in zip(Means, Stds, Params):
  print('{},{} with: {}'.format(mean, stdev, param))
print()

Model = KerasClassifier(build_fn = create_model,verbose = 0,batch_size = 40,epochs = 10)
Neuron1 = [4,8,16]
Neuron2 = [2,4,8]
Param_grids = dict(neuron1 = Neuron1,neuron2 = Neuron2)
Grid = GridSearchCV(estimator = Model,param_grid = Param_grids,cv = KFold(),verbose = 10)
Grid_Result = Grid.fit(X_standardized,Y)
print('Best : {}, using {}'.format(Grid_result.best_score_,Grid_result.best_params_))
print()

Means = Grid_result.cv_results_['mean_test_score']
Stds = Grid_result.cv_results_['std_test_score']
Params = Grid_result.cv_results_['params']

for mean, stdev, param in zip(Means, Stds, Params):
    print('{},{} with: {}'.format(Mean, Stdev, Params))
print()

Model = KerasClassifier(build_fn = create_model,verbose = 0,batch_size = 40,epochs = 10)

Model.fit(X_standardized,Y1)

Y_predict = Model.predict(X_standardized)

print(accuracy_score(Y1,Y_predict))
print()

Model = KerasClassifier(build_fn = create_model,verbose = 0)

Batch_size = [10,20,40]
Epochs = [10,50,100]
Learning_rate = [0.001,0.01,0.1]
Dropout_rate = [0.0,0.1,0.2]
Activation_function = ['softmax','relu','tanh','linear']
Init = ['uniform','normal','zero']
Neuron1 = [4,8,16]
Neuron2 = [2,4,8]

Param_grids = dict(batch_size = Batch_size,epochs = Epochs,learning_rate = Learning_rate,dropout_rate = Dropout_rate,activation_function = Activation_function,init = Init,neuron1 = Neuron1,neuron2 = Neuron2)

Grid = GridSearchCV(estimator = Model,param_grid = Param_grids,cv = KFold(),verbose = 10)
Grid_result = Grid.fit(X_standardized,Y1)
print('Best : {}, using {}'.format(Grid_result.best_score_,Grid_result.best_params_))
print()

Means = Grid_result.cv_results_['mean_test_score']
Stds = Grid_result.cv_results_['std_test_score']
Params = Grid_result.cv_results_['params']

for Mean, Stdev, Param in zip(Means, Stds, Params):
  print('{},{} with: {}'.format(Mean, Stdev, Param))
print()