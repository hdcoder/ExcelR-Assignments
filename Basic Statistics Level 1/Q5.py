Balls = ['R','R','G','G','G','B','B']

Two_Balls = []

length = len(Balls)

i = 0
j = 0
count = 0

for i in range(0,length):
    for j in range(0,length):
        if i!=j:
            Two_Balls.append([Balls[i],Balls[j]])

length1 = len(Two_Balls)

for k in Two_Balls:
    if k[0]=='B' and k[1]=='B':
        count+=1

print("Probability that none of the balls drawn is Blue is : " + str((length1-count)/length1))
