import pandas as pd
import numpy as np
from scipy import stats 
from scipy.stats import norm

Cutlets_Data = pd.read_csv("Cutlets.csv")
print(Cutlets_Data.head())
print()

Unit_A = pd.Series(Cutlets_Data["Unit A"])
Unit_B = pd.Series(Cutlets_Data["Unit B"])

P_Value = stats.ttest_ind(Unit_A, Unit_B)
print(P_Value)