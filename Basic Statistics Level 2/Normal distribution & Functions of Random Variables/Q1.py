
import math

mean = 45
SD = 8
T = 50

probability = (1/SD*math.sqrt(2*math.pi))*math.exp((-0.5)*(T-mean)*(T-mean)/(SD*SD))

print("Probability of the car not being ready before agreed time: ",probability)