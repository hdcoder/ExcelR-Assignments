Candies = [1,4,3,5,6,2]
Probabilities = [0.015,0.2,0.65,0.005,0.01,0.12]

length = len(Candies)
EV = 0

for i in range(0,length):
    EV = EV + Candies[i]*Probabilities[i]
    
print(EV)