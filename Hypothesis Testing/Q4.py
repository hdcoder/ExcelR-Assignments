
import pandas as pd
import scipy.stats as stats
from sklearn import preprocessing

COF = pd.read_csv("Customer+OrderForm.csv")

print(COF)

label_encoder = preprocessing.LabelEncoder()

COF["Phillippines"] = label_encoder.fit_transform(COF["Phillippines"])
COF["Indonesia"] = label_encoder.fit_transform(COF["Indonesia"])
COF["Malta"] = label_encoder.fit_transform(COF["Malta"])
COF["India"] = label_encoder.fit_transform(COF["India"])

print(COF)

fvalue, pvalue = stats.f_oneway(COF["Phillippines"],COF["Indonesia"],COF["Malta"],COF["India"])

print(fvalue, pvalue)
