import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

Bank = pd.read_csv("bank-full.csv")

X = Bank.iloc[:,0:16]
Y = Bank["Target"]

le = LabelEncoder()

X["job"] = le.fit_transform(X["job"])
X["marital"] = le.fit_transform(X["marital"])
X["education"] = le.fit_transform(X["education"])
X["default"] = le.fit_transform(X["default"])
X["housing"] = le.fit_transform(X["housing"])
X["loan"] = le.fit_transform(X["loan"])
X["contact"] = le.fit_transform(X["contact"])
X["month"] = le.fit_transform(X["month"])
X["poutcome"] = le.fit_transform(X["poutcome"])
Y = le.fit_transform(Y)

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.50, random_state = 0)

classifier = LogisticRegression(random_state = 0)
classifier.fit(X_train, Y_train)
Y_pred = classifier.predict(X_test)

print(Y_pred)

cm = confusion_matrix(Y_test, Y_pred)
 
print ("Confusion Matrix : \n", cm)
print ("Accuracy : ", accuracy_score(Y_test, Y_pred))