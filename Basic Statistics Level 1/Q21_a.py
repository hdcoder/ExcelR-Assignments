import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm
import statistics

df = pd.read_csv("wc-at.csv")

x_axis = df["AT"]

mean = statistics.mean(x_axis)
sd = statistics.stdev(x_axis)

fig, ax = plt.subplots(figsize =(10, 7))
ax.hist(x_axis, bins = 20)

plt.plot(x_axis, norm.pdf(x_axis, mean, sd))
plt.show()
