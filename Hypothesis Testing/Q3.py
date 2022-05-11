
import pandas as pd
import scipy.stats as stats

Buyer = pd.read_csv("BuyerRatio.csv")

a = Buyer["East"]
b = Buyer["West"]
c = Buyer["North"]
d = Buyer["South"]

fvalue, pvalue = stats.f_oneway(a,b,c,d)

print(fvalue, pvalue)
