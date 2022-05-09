
import numpy as np
from scipy.stats import norm

mean = 100
SD = 20

normal = norm(loc = mean , scale = SD)

probability1 = normal.cdf(48.5)
probability2 = normal.cdf(151.5)

probability = probability2 - probability1

print(probability)

