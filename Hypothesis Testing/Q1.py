
import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

Cutlets = pd.read_csv("Cutlets.csv")

Unit_A = Cutlets["Unit A"]
Unit_B = Cutlets["Unit B"]

Unit_A_mean = np.mean(Unit_A)
Unit_B_mean = np.mean(Unit_B)

print("Unit A mean value :",Unit_A_mean)
print("Unit B mean value :",Unit_B_mean)

Unit_A_std = np.std(Unit_A)
Unit_B_std = np.std(Unit_B)

print("Unit A std value :",Unit_A_std)
print("Unit B std value :",Unit_B_std)

ttest,pval = ttest_ind(Unit_A,Unit_B)
print("P-value :",pval)

if pval < 0.05 :
  print("We reject null hypothesis")
else:
  print("We accept null hypothesis")