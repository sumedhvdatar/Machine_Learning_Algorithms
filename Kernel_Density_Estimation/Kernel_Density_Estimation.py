#Author: Sumedh Vilas Datar
# This file contains implementation of Kernel Density function
# for 1 Dimension and 2 Dimension.
import numpy as np
from matplotlib import pyplot as plt
bins = None
from mpl_toolkits.mplot3d import Axes3D

def generae_one_dimensional_data(mu,sigma,number_of_samples):
    s = np.random.normal(mu, sigma,number_of_samples)
    return s

def generate_two_dimensinal_data(mu1,cov,number_of_samples):
    samples = np.random.multivariate_normal(mu1, cov,number_of_samples)
    return samples

def calculate_k_u(data_point,x,h):
    result = (x - data_point)/h
    if abs(result) <= 0.5:
        return 1
    else:
        return 0

def calculate_k(data,x,h):
    sum = 0
    for s in data:
        sum = sum + calculate_k_u(s,x,h)
    return sum

def myKDE(samples,X,h):
    probabilies = []
    x_values = []
    bin = 0
    while (X < upper_range):
        K = calculate_k(samples, X, h)
        probability_x = K / (NUMBER_OF_SAMPLES * h)
        probabilies.append(probability_x)
        x_values.append(X)
        X = X + delta
        bin = bin + 1
    return probabilies,x_values


def calculate_k_u2(data_point,x,h):
    prod = 1
    i=0
    for x1 in x:
        result = (x1 - data_point[i])/h
        if abs(result) <= 0.5:
            prod = prod * 1
        else:
            prod = prod * 0
        i = i + 1
    return prod

def calculate_k_2(data,x,h):
    sum = 0
    for s in data:
        sum = sum + calculate_k_u2(s, x, h)
    return sum

def myKDE2(samples,X,h):
    probabilies = []
    x_values = []
    while X[0] < upper_range_2[0]:
        K = calculate_k_2(samples,X,h)
        probability_x = K / (NUMBER_OF_SAMPLES * h)
        probabilies.append(probability_x)
        x_values.append(X)
        X = [X[0]+delta,X[1]+delta]
    return probabilies,x_values



dimension = 2
q = "q4"
if dimension == 1 and q == "q2":
    MU = 5
    SIGMA = 1
    NUMBER_OF_SAMPLES = 1000
    lower_Range = -1
    upper_range = 10
    delta = 0.01
    h = 1 #You can change all h values here for q2
    samples = generae_one_dimensional_data(MU,SIGMA,NUMBER_OF_SAMPLES)
    plt.title("Histogram of data")
    plt.ylabel("Number of times")
    plt.hist(samples)
    X = lower_Range
    probabilies = []
    probabilies,x_values = myKDE(samples,X,h)
    fig, ax = plt.subplots()
    plt.title("Estimated density when h = " + str(h))
    plt.xlabel("X Values")
    plt.ylabel("Probabilities")
    # plt.plot(x_values,probabilies,color = "red")
    # plt.plot(samples,color = "green")

    plt.plot(x_values,probabilies,color = "red")
    plt.show()

elif dimension == 1 and q == "q3":
    MU = 5
    SIGMA = 1
    MU2 = 0
    SIGMA2 = 0.2
    NUMBER_OF_SAMPLES = 1000
    lower_Range = -1
    upper_range = 10
    delta = 0.01
    h = 1
    samples1 = generae_one_dimensional_data(MU,SIGMA,NUMBER_OF_SAMPLES)
    samples2 = generae_one_dimensional_data(MU2,SIGMA2,NUMBER_OF_SAMPLES)
    samples = np.concatenate((samples1,samples2))
    print(samples.shape)
    plt.title("Histogram of data")
    plt.ylabel("Number of times")
    plt.hist(samples)
    X = lower_Range
    probabilies = []
    probabilies,x_values = myKDE(samples,X,h)
    fig, ax = plt.subplots()
    plt.title("Estimated density when h = " + str(h))
    plt.xlabel("X Values")
    plt.ylabel("Probabilities")
    # plt.plot(x_values,probabilies,color = "red")
    # plt.plot(samples,color = "green")

    plt.plot(x_values,probabilies,color = "red")
    plt.show()


elif dimension == 2 and q == "q4":
    MU1 = [1,0]
    MU2 = [0,1.5]
    NUMBER_OF_SAMPLES = 500
    lower_Range = [-3,-3]
    upper_range_2 = [5,5]
    delta = 0.01
    h = 1
    COV = [[0.9,0.4],[0.4,0.9]]
    samples1 = generate_two_dimensinal_data(MU1,COV, NUMBER_OF_SAMPLES)
    samples2 = generate_two_dimensinal_data(MU2,COV,NUMBER_OF_SAMPLES)
    samples = np.concatenate((samples1,samples2))
    probabilies, x_values = myKDE2(samples,lower_Range, h)
    # print(samples.shape)
    fig = plt.figure()
    ax = fig.add_subplot(111,projection='3d')
    x_values = np.asarray(x_values)
    ax.plot(x_values[:,0],x_values[:,1],probabilies, c="r")
    plt.title("Estimated density when h = " + str(h))
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.show()


