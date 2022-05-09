
import numpy as np
from scipy.stats import norm

mean = 38
SD = 6

normal = norm(loc=mean , scale = SD)

probability = normal.cdf(30)

employees = probability*400

if employees>35 and employees<37:
    print("True")
else:
    print("False")