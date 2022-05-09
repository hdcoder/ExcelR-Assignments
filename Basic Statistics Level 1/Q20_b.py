import pandas as pd
import numpy as np

df = pd.read_csv("Cars.csv")
MPG = df["MPG"]

length = len(MPG)
count = 0

for i in MPG:
    if i>20 and i<50:
        count += 1

print("Probability of MPG between 20 and 50 is : " + str(count/length))