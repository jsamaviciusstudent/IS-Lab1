# Lab 1
import os
import numpy as np
#  1) import data.txt (Image data already calcualted)
# Build 2x5 matrix
# Color | Roundness | Apple/Pear

# Pagal du objektus reikia apmokyti perceptrona, kad atpazintu obuoli/kriause
# First algorith:
# Pagal estimated (data.txt) features paemiau pirmuju 3 obuoliu ir 2 kriausiu duomenis
# Color / Roundness.
# Apmokiau viena perceptrona, kuris atpazintu siuos features ir nustaciau pavyzdini perceptrona.
# Second algorithm:
# 
# Tada apmokymui paemiui visus obuolius/kriauses ir ejau pro iteracijas,
# Lygindamas su "desired response" kol total_error taps 0
#
data = np.loadtxt("Data.txt", dtype='d', delimiter=',')
print(data)

# Extract First 3 apples and First 2 Pears
## HSV values
hsv_apples = data[0:3, [0]] # first 3 apples
hsv_pears = data [10:12, [0]] # first 2 pears
print('First 3 apples: ', hsv_apples)
print('First 2 pears:', hsv_pears)

# combine
hsv_array = np.concatenate((hsv_apples, hsv_pears)) #x1
print('Combined HSV array', hsv_array)

## Metric values

metric_apples = data[0:3, [1]] # first 3 apples metrics
metric_pears = data[10:12, [1]] # first 2 pears metrics

metric_array = np.concatenate((metric_apples, metric_pears)) # x2
print('Combined metric array: \n', metric_array)


metric_hsv_matrix = np.vstack((hsv_array.T, metric_array.T))  # Transpose to match the shape
print('Combined 2x5 matrix (HSV and Metric):\n', metric_hsv_matrix)

T = [1, 1, 1, -1, -1]

# Generating random initial values of w1, w2 and b
w1 = np.random.randn()
w2 = np.random.randn()
b = np.random.randn()

# Weight = x1 * w1 + x2 * w2 + b

Weights = []
Errors = []

def per_output(hsv_value, metric_value, w1, w2, b):
    v = hsv_value * w1 + metric_value * w2 + b
    return 1 if v > 0 else -1

for i in range(len(hsv_array)):

    weight = per_output(hsv_array[i], metric_array[i], w1, w2, b)

    Weights.append(weight) # Not really needed probably, but saving it to log/print incase mistakes


    e = T[i] - weight

    Errors.append(e)


print('--------')
print("Original Weights: ",Weights)
print("Original Errors: ", Errors)
print('--------')

# Total error count


total_error = abs(Errors[0]) + abs(Errors[1]) + abs(Errors[2]) + abs(Errors[3]) + abs(Errors[4]) 

print("Original total_error: ", total_error)



### Training/Testing algorithm ###

## Testing how good are updated parameters on all examples used for training: 

new_weights = [0] * 6
new_errors = [0] * 6
iteration = 1
while total_error !=  0:
    print('-~- Start of new iteration -~-')
    print('Iteration number: ', iteration)
    print(f'w1 = {w1}, w2 = {w2}, b = {b}')
    print('Errors: ', new_errors)
    print('Weights: ', new_weights)
    print('Total errors: ', total_error)
    print(' -~- -~- -~- -~- -~- -~-')
    for i in range(len(hsv_array)):
        w1 = w1 + e * 0.1 * hsv_array[i]
        w2 = w2 + e * 0.1 * metric_array[i]
        b = b + 0.1 * e * 1
      # Counting weighted sum:
        
        v = per_output(hsv_array[i], metric_array[i], w1, w2, b)

        new_weights[i] = v
    
        e = T[i] - new_weights[i]

        new_errors[i] = e


       # Comparing with last example:


    iteration += 1
    total_error = abs(new_errors[0]) + abs(new_errors[1]) + abs(new_errors[2]) + abs(new_errors[3]) + abs(new_errors[4]) 


print('-!!- FINAL iteration -!!- ')
print('Iteration number: ', iteration)
print(f'w1 = {w1}, w2 = {w2}, b = {b}')
print('Errors: ', new_errors)

print('Weights: ', new_weights)
print('Total errors: ', total_error)
print('Final Total errors: ', total_error)

        
