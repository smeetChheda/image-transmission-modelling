import random
import numpy as np
import matplotlib.pyplot as plt

# These values come from running without noise
# Simpler to store than to rerun everytime

# key = R, value = centers for given R no noise
starting_centers = {
    1: [0.7716627813067206, -0.8210400219407225],
    2: [-1.4971801750896505, -0.4377230374030127, 0.4790989777059966, 1.5417613817364066],
    3: [0.7316182068055997, 2.153085185897407, -0.24165406491113614, -0.7538056243753133, 0.23475773954183032, 1.3152343609881862, -1.3530664703560213, -2.2036399192609886],
    4: [1.9483584182188136, 0.3367954916218906, -1.6114850920332418, 1.5242821615586253, -0.9116789622394401, -2.1399998171175887, 2.5774822809391686, 1.1712453973980126, 0.5916346919872297, -0.1542563669034436, -2.8700965620048433, -1.2324719961501283, -0.3981074524934233, 0.8625246639369076, 0.08469569485902159, -0.6493831531752919]
}

def DecimalToBinary (dec, store, R):
    store = str((dec%2)) + store
    
    if (dec == 0 or dec == 1):
        # makes them all the same length
        for i in range(R + 1 - len(store)):
            store = '0' + store
        return store
    return DecimalToBinary(dec//2, store, R)

def condProb (a, b, eps):
    if len(a) != len(b):
        print('two binary values should have same length')
        return -1

    output = 1
    for i in range(len(a)):
        if a[i] == b[i]:
            output *= 1-eps
        else:
            output *= eps
    return output


def lloyds_noise (X, centers, R, distortion_list, iteration, eps):
    partitions = []
    distortion_sum = 0
    num_training_vals = len(X)
    n = 2**R

    for i in range(n):
        partitions.append([])

# super slow - embedded for loops
    for x in X:
        sums = [0]*n

        for j in range(n):
            j_bin = DecimalToBinary(j, '', R)

            for i in range(n):
                i_bin = DecimalToBinary(i, '', R)

                # switched this to sums[i] since we keep j constant and vary i
                # and keep track of different i options
                # not sure if this is correct
                sums[i] += condProb(j_bin, i_bin, eps) * (x-centers[j])**2

            
        # arb large number
        min_val = 1000
        
        for group in range(n):
            if sums[group] <= min_val:
                part_group = group
                min_val = sums[group]
            
        partitions[part_group].append(x)
        distortion_sum += sums[part_group]
    
    distortion_list.append(distortion_sum/num_training_vals)
# use a stopping criteria - see Julien's thesis
    if(iteration > 30):
        return centers, partitions, distortion_list
    
    # calculates new centers - not sure if this is correct
    for j in range(n):
        numerator_sum = 0
        denominator_sum = 0
        j_bin = DecimalToBinary(j, '', R)
        for i in range(n):
            i_bin = DecimalToBinary(i, '', R)
            conditional_probability = condProb(j_bin, i_bin, eps)
            denominator_sum += conditional_probability*len(partitions[i])/num_training_vals
            inner_sum = 0
            for xl in partitions[i]:
                inner_sum += xl

            numerator_sum += conditional_probability*inner_sum/num_training_vals

        centers[j] = numerator_sum/denominator_sum

    return lloyds_noise(X, centers, R, distortion_list, iteration + 1, eps)


X = np.random.normal(0, 1, 10000)
normal_list = []
for num in X:
    normal_list.append(num)

# eps = 0.1
eps_arr = [0, 0.001, 0.01, 0.1, 0.5]
R = 1

dists = []
for eps in eps_arr:
    centersNoise, partitionsNoise,distortion_listNoise = lloyds_noise(normal_list, starting_centers[R], R, [], 1, eps)
    print(eps)
    print(centersNoise)
    print('\n')
    dists.append(distortion_listNoise[-1])

plt.plot(eps_arr, dists, '.')
plt.xlabel("Error")
plt.ylabel("Distortion")
plt.title("Distortion Error Graph - R = 2")
plt.show()


# centersNoise, partitionsNoise,distortion_listNoise = lloyds_noise(normal_list, starting_centers[R], R, [], 1, eps)
# print(centersNoise)

# x = [i+1 for i in range(len(distortion_listNoise))]
# plt.plot(x, distortion_listNoise, label = 'N = ' + str(2**R))
    
# plt.xlabel("Iteration")
# plt.ylabel("Distortion")
# plt.title("Distortion for 4 Level Quantization")
# plt.legend()
# plt.show()

# for R in range(1, 4):
#     centersNN, partitionsNN, distortion_listNN = lloyds_noise(normal_list, starting_centers[R], R, [], 1, eps)
#     print('R = ' + str(R) + '\n')
#     print(centersNN)
#     print('\n')


#     x = [i+1 for i in range(len(distortion_listNN))]
#     plt.plot(x, distortion_listNN, label = 'N = ' + str(2**R))
    
# plt.xlabel("Iteration")
# plt.ylabel("Distortion")
# plt.title("Distortion for N Level Quantization With Noise")
# plt.legend()
# plt.show()   
    