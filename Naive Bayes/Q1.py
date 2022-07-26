import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB as MB
from sklearn.feature_extraction.text import TfidfTransformer

def split_into_words(i):
    return (i.split(" "))

Salary_Data = pd.read_csv("Salary_Data_Test.csv")
print(Salary_Data.head())
print()

print(Salary_Data.isnull().sum())
print()

print(Salary_Data.info())
print()

Salary_Data_train,Salary_Data_test = train_test_split(Salary_Data, test_size=0.3)

Data_matrix = CountVectorizer(analyzer=split_into_words).fit(Salary_Data.Salary)
print(Data_matrix)
print()

Data_matrix_all = Data_matrix.transform(Salary_Data.Salary)
print(Data_matrix_all.shape)
print()

Train_Data_matrix = Data_matrix.transform(Salary_Data_train.Salary)
print(Train_Data_matrix.shape)
print()

Test_Data_matrix = Data_matrix.transform(Salary_Data_test.Salary)
print(Test_Data_matrix.shape)
print()

Classifier_mb = MB()
Classifier_mb.fit(Train_Data_matrix,Salary_Data_train.workclass)

Train_Pred_m = Classifier_mb.predict(Train_Data_matrix)
Accuracy_Train_m = np.mean(Train_Pred_m==Salary_Data_train.workclass)

Test_Pred_m = Classifier_mb.predict(Test_Data_matrix)
Accuracy_Test_m =np.mean(Test_Pred_m==Salary_Data_test.workclass)

print("Accuracy of Train Data : ",Accuracy_Train_m)
print("Accuracy of Test Data : ",Accuracy_Test_m)
print()

Tfidf_transformer = TfidfTransformer().fit(Data_matrix_all)

Train_tfidf = Tfidf_transformer.transform(Train_Data_matrix)
print("train_tfidf.shape : ",Train_tfidf.shape)
print()

Test_tfidf = Tfidf_transformer.transform(Test_Data_matrix)
print("test_tfidf.shape : ",Test_tfidf.shape)
print()

Classifier_mb = MB()
Classifier_mb.fit(Train_tfidf,Salary_Data_train.workclass)

Train_Predict_mb = Classifier_mb.predict(Train_tfidf)
Accuracy_Train_mb = np.mean(Train_Predict_mb == Salary_Data_train.workclass)

Test_Predict_mb = Classifier_mb.predict(Test_tfidf)
Accuracy_Test_mb = np.mean(Test_Predict_mb == Salary_Data_test.workclass)

print("Accuracy of Train Data : ",Accuracy_Train_mb)
print("Accuracy of Test Data : ",Accuracy_Test_mb)