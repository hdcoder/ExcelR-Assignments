import pandas as pd
import numpy as np
from scipy import stats 
from scipy.stats import norm
from scipy.stats import chi2_contingency

OrderForm_Data = pd.read_csv("Customer+OrderForm.csv")
print(OrderForm_Data)
print()

PH = OrderForm_Data.Phillippines.value_counts()
IN = OrderForm_Data.India.value_counts()
MA = OrderForm_Data.Malta.value_counts()
INDO = OrderForm_Data.Indonesia.value_counts()

Obs_Array = np.array([PH,IN,MA,INDO])
print(Obs_Array)

stat, p, dof, expected = chi2_contingency(Obs_Array)

print("P value : ",p)

if p < 0.05 :
    print("We accept Ha & Means are not equal")
else:
    print("We accept Ho & Means are equal")