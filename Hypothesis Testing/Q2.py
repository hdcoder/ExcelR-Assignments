import pandas as pd
import numpy as np
from scipy import stats 
from scipy.stats import norm

LabTAT_Data = pd.read_csv("labTAT.csv")
print(LabTAT_Data)
print()

P_Value = stats.f_oneway(LabTAT_Data.iloc[:,0],LabTAT_Data.iloc[:,1],LabTAT_Data.iloc[:,2],LabTAT_Data.iloc[:,3])
print(P_Value)