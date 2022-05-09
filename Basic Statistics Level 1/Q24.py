from scipy.stats import norm
  
mean = 260
std = 90
  
probability_cdf = norm.cdf(261,loc=mean, scale=std)

print("The probability of Bulbs not having average life of more than 260 : " + str(probability_cdf))