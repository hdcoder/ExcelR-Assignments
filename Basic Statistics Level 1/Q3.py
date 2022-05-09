Three_coins = ["HHH","HHT","HTT","HTH","THH","TTH","THT","TTT"]
length = len(Three_coins)
count = 0

for i in Three_coins:
    result = list(i)
    heads = 0
    tails = 0
    for j in result:
        if j=='H':
            heads += 1
        else:
            tails += 1
    if heads==2 and tails==1:
        count += 1

print("Probability of Two heads and One Tail is : " + str(count/length))