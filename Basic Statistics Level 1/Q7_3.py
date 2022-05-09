import pandas as pd
import numpy as np
import statistics as stats

df = pd.read_csv("Q7.csv")
mean = np.mean(df.Weigh)
median = np.median(df.Weigh)
mode = stats.mode(df.Weigh)
std = np.std(df.Weigh)
var = np.var(df.Weigh)
minimum = min(df.Weigh)
maximum = max(df.Weigh)


print("Mean : " + str(mean))
print("Median : " + str(median))
print("Mode : " + str(mode))
print("Standard Deviation : " + str(std))
print("Variance : " + str(var))
print("Range : " + str(minimum) + "," + str(maximum))
