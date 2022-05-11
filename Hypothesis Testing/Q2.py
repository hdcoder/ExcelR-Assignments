
import pandas as pd
import scipy.stats as stats

labs = pd.read_csv("LabTAT.csv")

lab1 = labs["Laboratory 1"]
lab2 = labs["Laboratory 2"]
lab3 = labs["Laboratory 3"]
lab4 = labs["Laboratory 4"]

Fvalue, Pvalue = stats.f_oneway(lab1,lab2,lab3,lab4)

print(Fvalue, Pvalue)

if Pvalue < 0.05:
    print("We accept null hypothesis")
else:
    print("We dont accept null hypothesis")