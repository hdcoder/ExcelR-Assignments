from scipy import stats

alpha1 = 0.90
alpha2 = 0.94
alpha3 = 0.60

Z_critical = stats.norm.ppf(1 - alpha1)
print(Z_critical)

Z_critical = stats.norm.ppf(1 - alpha2)
print(Z_critical)

Z_critical = stats.norm.ppf(1 - alpha3)
print(Z_critical)
