import pandas as pd
import numpy as np
import statistics as stats

df = pd.read_csv("Q7.csv")
mean = np.mean(df.Points)
median = np.median(df.Points)
mode = stats.mode(df.Points)
std = np.std(df.Points)
var = np.var(df.Points)
minimum = min(df.Points)
maximum = max(df.Points)

print("Mean : " + str(mean))
print("Median : " + str(median))
print("Mode : " + str(mode))
print("Standard Deviation : " + str(std))
print("Variance : " + str(var))
print("Range : " + str(minimum) + "," + str(maximum))
