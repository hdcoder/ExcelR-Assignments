
import numpy as np
from scipy.stats import norm

mean = 38
SD = 6

normal = norm(loc=mean , scale = SD)

probability1 = 1-normal.cdf(44)

print(probability1)

probability2 = normal.cdf(44) - normal.cdf(38)

print(probability2)

if probability1 > probability2:
    print("True")
else:
    print("False")