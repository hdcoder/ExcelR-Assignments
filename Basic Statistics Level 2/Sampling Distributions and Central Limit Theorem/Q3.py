import numpy as np
from scipy.stats import norm

mean = 50
SD = 40

normal = norm(loc = mean , scale = SD)

probability1 = normal.cdf(45)
probability2 = normal.cdf(55)

probability = probability2 - probability1

print(1-probability)