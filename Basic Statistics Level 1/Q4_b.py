Two_dice = [[1,1],[1,2],[1,3],[1,4],[1,5],[1,6],
[2,1],[2,2],[2,3],[2,4],[2,5],[2,6],
[3,1],[3,2],[3,3],[3,4],[3,5],[3,6],
[4,1],[4,2],[4,3],[4,4],[4,5],[4,6],
[5,1],[5,2],[5,3],[5,4],[5,5],[5,6],
[6,1],[6,2],[6,3],[6,4],[6,5],[6,6]]

length = len(Two_dice)
count = 0

for i in Two_dice:
    sum = 0
    for j in i:
        sum += j

    if sum <= 4 :
        count += 1

print("Probability that sum is less than or equal to 4 is :" + str(count/length))


