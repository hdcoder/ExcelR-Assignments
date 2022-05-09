import numpy as np
from scipy.stats import t

x = np.random.normal(size=2000)
mean = 200
SD = 30
dof = len(x)-1
Confidence = 0.98

t_crit = np.abs(t.ppf((1-Confidence)/2,dof))

print([mean-SD*t_crit/np.sqrt(len(x)), m+SD*t_crit/np.sqrt(len(x))])