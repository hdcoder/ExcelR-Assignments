import pandas as pd
import numpy as np
import statistics as stats

df = pd.read_csv("Q7.csv")
mean = np.mean(df.Score)
median = np.median(df.Score)
mode = stats.mode(df.Score)
std = np.std(df.Score)
var = np.var(df.Score)
minimum = min(df.Score)
maximum = max(df.Score)

print("Mean : " + str(mean))
print("Median : " + str(median))
print("Mode : " + str(mode))
print("Standard Deviation : " + str(std))
print("Variance : " + str(var))
print("Range : " + str(minimum) + "," + str(maximum))