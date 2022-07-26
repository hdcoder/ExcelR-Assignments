import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report,confusion_matrix
from sklearn.model_selection import cross_val_score

Glass_Data = pd.read_csv("Glass.csv")
Glass_Data.head()

print(Glass_Data.info())
print()

Scaler = StandardScaler()
Scaler.fit(Glass_Data.drop("Type",axis=1))

Scaled_Features = Scaler.transform(Glass_Data.drop("animal name",axis=1))
print(Scaled_Features)
print()

Glass_Data_Feat = pd.DataFrame(Scaled_Features,columns=Glass_Data.columns[:-1])
print(Glass_Data_Feat.head())
print()

X_train,X_test,Y_train,Y_test = train_test_split(Scaled_Features, Glass_Data["type"], test_size=0.3) 

sns.pairplot(Glass_Data,hue="Type")

KNN = KNeighborsClassifier(n_neighbors = 1)
KNN.fit(X_train,Y_train)

Pred = KNN.predict(X_test)

print(confusion_matrix(Y_test,Pred))
print()

print(classification_report(Y_test,Pred))
print()

accuracy_rate = []

for i in range(1,40):

    KNN = KNeighborsClassifier(n_neighbors=i)
    Score = cross_val_score(KNN,Glass_Data_Feat,Glass_Data["type"],cv=10)
    accuracy_rate.append(Score.mean())

Error_Rate = []

for i in range(1,40) :

    knn = KNeighborsClassifier(n_neighbors=i)
    Score = cross_val_score(KNN,Glass_Data_Feat, Glass_Data['Type'],cv=10)
    Error_Rate.append(1-Score.mean())

plt.figure(figsize = (10,6))
plt.plot(range(1,40),accuracy_rate,color="blue",linestyle="dashed",marker = "o", markerfacecolor = "red",markersize=10 )
plt.title('Error Rate vs. K Value')
plt.xlabel('K')
plt.ylabel('Error Rate')

Error_Rate = []

for i in range(1,40):
    
    KNN = KNeighborsClassifier(n_neighbors=i)
    KNN.fit(X_train,Y_train)
    Pred_i = KNN.predict(X_test)
    Error_Rate.append(np.mean(Pred_i != Y_test))

plt.figure(figsize=(10,6))

plt.plot(range(1,40),accuracy_rate,color='blue', linestyle='dashed', marker='o',markerfacecolor='red', markersize=10)

knn = KNeighborsClassifier(n_neighbors=1)

KNN.fit(X_train,Y_train)
pred = KNN.predict(X_test)

print('WITH K=1')
print()
print(confusion_matrix(Y_test,Pred))
print()
print(classification_report(Y_test,Pred))