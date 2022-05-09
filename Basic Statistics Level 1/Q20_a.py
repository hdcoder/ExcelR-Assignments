import pandas as pd
import numpy as np

df = pd.read_csv("Cars.csv")
MPG = df["MPG"]

length = len(MPG)
count = 0

for i in MPG:
    if i<40:
        count += 1

print("Probability of MPG lesser than 40 is : " + str(count/length))