from scipy import stats
import math

alpha1 = 0.95
alpha2 = 0.96
alpha3 = 0.99

n = 25

t_critical = (stats.norm.ppf(1 - alpha1))/math.sqrt(n)
print(t_critical)

t_critical = (stats.norm.ppf(1 - alpha2))/math.sqrt(n)
print(t_critical)

t_critical = (stats.norm.ppf(1 - alpha3))/math.sqrt(n)
print(t_critical)
