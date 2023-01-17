import random
import numpy as np
import matplotlib.pyplot as plt

def condProb (a, b, eps):
    return 1 - eps if a==b else eps

def lloyds_noise (X, centers, distortion_list, iteration, eps):
    partitions = [[],[]]
    distortion_sum = 0
    n = len(X)

    if centers == []:
        for i in range(2):
            centers.append(random.choice(X))
    
    for x in X:
        left_sum = 0
        right_sum = 0

        for j in range(2):
            left_sum += condProb(j, 0, eps) * (x-centers[j])**2
            right_sum += condProb(j, 1, eps)*(x-centers[j])**2

        if left_sum <= right_sum:
            partitions[0].append(x)
        else:
            partitions[1].append(x)

    
    distortion_sum = 0
    # len(parititions) = 2 - could just use 2
    for i in range(len(partitions)):
        for xl in partitions[i]:
            for j in range(2):
                distortion_sum += condProb(j, i, eps)*(xl - centers[j])**2
    
    distortion_list.append(distortion_sum/n)

     # up this to about 200
    if(iteration > 100):
        return centers, partitions, distortion_list
    
    # new_centers = []
    for j in range(2):
        numerator_sum = 0
        denominator_sum = 0
        for i in range(2):
            denominator_sum += condProb(j, i, eps)*len(partitions[i])/n
            inner_sum = 0
            for xl in partitions[i]:
                inner_sum += xl

            # not sure if should be using n or len(partitions[i])
            numerator_sum += condProb(j, i, eps)*inner_sum/n

        centers[j] = numerator_sum/denominator_sum

    return lloyds_noise(X, centers, distortion_list, iteration + 1, eps)


X = np.random.normal(0, 1, 10000)
normal_list = []
for num in X:
    normal_list.append(num)
    
centers, partitions, distortion_list = lloyds_noise(normal_list, [], [], 1, 0.1)

print(centers)
print('\n')
print(distortion_list[-1])
print('\n')
print(partitions[0][:10])
print('\n')
print(partitions[1][:10])
print('\n')

x = [i+1 for i in range(len(distortion_list))]

plt.scatter(x, distortion_list)
plt.title("Distortion graph")
plt.show()    
    