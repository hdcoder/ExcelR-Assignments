import numpy as np
import statistics as stats
import matplotlib.pyplot as plt
import seaborn as sns

X = [24.23,25.53,25.41,24.14,29.62,28.25,25.81,24.39,40.26,32.95,91.36,25.99,39.42,26.71,35.00]

fig, ax = plt.subplots(2)

ax[0].hist(X, bins = 20)

ax[1] = sns.boxplot(X)

mean = stats.mean(X)
std = stats.stdev(X)
var = stats.variance(X)

print(mean)
print(std)
print(var)