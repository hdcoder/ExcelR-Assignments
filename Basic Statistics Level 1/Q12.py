import numpy as np
from matplotlib import pyplot as plt

Scores = [34,36,36,38,38,39,39,40,40,41,41,41,41,42,42,45,49,56]

mean = np.mean(Scores)
median = np.median(Scores)
std = np.std(Scores)
var = np.var(Scores)

fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(Scores , bins = 20)

plt.show()

print("Mean : " + str(mean))
print("Median : " + str(median))
print("Standard Deviation : " + str(std))
print("Variance : " + str(var))
