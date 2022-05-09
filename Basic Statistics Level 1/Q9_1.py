import pandas as pd
from scipy import stats

df = pd.read_csv("Q9_a.csv")

skew_speed = stats.skew(df.speed , bias=False)
kurtosis_speed = stats.kurtosis(df.speed , bias=False)
print(skew_speed)
print(kurtosis_speed)

skew_dist = stats.skew(df.dist , bias=False)
kurtosis_dist = stats.kurtosis(df.dist , bias=False)
print(skew_dist)
print(kurtosis_dist)