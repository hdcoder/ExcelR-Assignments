import pandas as pd
import numpy as np
from scipy import stats 
from scipy.stats import norm
from scipy.stats import chi2_contingency

BuyerRatio_Data = pd.read_csv("BuyerRatio.csv")
print(BuyerRatio_Data)

print("The level of significance is 95%")

E = BuyerRatio_Data.East
W = BuyerRatio_Data.West
N = BuyerRatio_Data.North
S = BuyerRatio_Data.South

Obs_Array = np.array([E,W,N,S])
print(Obs_Array)

stat, p, dof, expected = chi2_contingency(Obs_Array)

print("P value : ",p)

if p < 0.05 :
    print("We accept Ha & Means are not equal")
else:
    print("We accept Ho & Means are equal")