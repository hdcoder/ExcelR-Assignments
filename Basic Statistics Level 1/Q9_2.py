import pandas as pd
from scipy import stats

df = pd.read_csv("Q9_b.csv")
skew_SP = stats.skew(df.SP , bias=False)
kurtosis_SP = stats.kurtosis(df.SP , bias=False)
print(skew_SP)
print(kurtosis_SP)

skew_WT = stats.skew(df.WT , bias=False)
kurtosis_WT = stats.kurtosis(df.WT , bias=False)
print(skew_WT)
print(kurtosis_WT)
