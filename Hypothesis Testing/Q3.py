import pandas as pd
from scipy.stats import chi2_contingency

Buyer = pd.read_csv("BuyerRatio.csv")

Buyer = Buyer.iloc[0:,1:]

print(Buyer)

stat, p, dof, expected = chi2_contingency(Buyer)
  
alpha = 0.05

print("p value is " + str(p))

if p <= alpha:
    print('Reject Null Hypothesis.The ratios are not similar.')

else:
    print('Accept null Hypothesis.The ratios are similar.')