#Author : Sumedh Vilas Datar
#This file contains the implementation details of Logisitic Regression.
import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix
import itertools

def generate_samples(mu1,cov,number_of_samples):
    """
    This method will generate the samples given input parameters.
    :param mu1:
    :param cov:
    :param number_of_samples:
    :return: samples
    """
    samples = np.random.multivariate_normal(mu1, cov,number_of_samples)
    return samples

def display(samples11,samples22,weights,training_data,learning_rate):
    x_values = [np.min(training_data[:, 1] ), np.max(training_data[:, 2] )]
    y_values = - (weights[0] + np.dot(weights[1], x_values)) / weights[2]
    fig, ax = plt.subplots()
    ax.scatter(samples11[:,0],samples11[:,1],color = "black")
    ax.scatter(samples22[:,0],samples22[:,1],color="red")
    ax.plot(x_values, y_values, label='Decision Boundary')
    plt.title('1000 Test samples with learning rate = '+str(learning_rate))

    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.show()

def display_train_error(e,i,learning_rate):
    fig, ax = plt.subplots()
    plt.title('Training Error when learning rate = '+str(learning_rate))
    plt.plot(i,e, label='Train Error')
    plt.ylabel('Train Error')
    plt.xlabel('Epochs')
    plt.show()

def display_gradient_curve(g,i,learning_rate):
    fig, ax = plt.subplots()
    plt.title('Gradient Curve when learning rate = '+str(learning_rate))
    plt.plot(i,g, label='Gradient Curve')
    plt.ylabel('Gradients')
    plt.xlabel('Number of Epochs')
    plt.show()

def logistic_regression_batch(x_train,y,learning_rate):

    weights = [1,1,1]
    training_error = []
    iterations = []
    grad_list = []
    for  i in range(0,100000):
        output = x_train.dot(weights)
        predicted = 1 / (1 + np.exp(-output))
        cost = -np.sum((y * np.log(predicted)) + (((1 - y) * np.log(1 - predicted)))) / len(y)
        training_error.append(cost)
        iterations.append(i)
        numerator = np.dot(x_train.T, predicted - y)
        gradient = numerator / len(y)
        grad_list.append(np.linalg.norm((gradient), ord=1))
        new_gradient = gradient * learning_rate
        weights = weights - new_gradient

        if np.linalg.norm((gradient), ord=1) < 0.001:
            # print("breaking")
            # print(i)
            break
    print("Number of iteration = "+str(i))
    return weights,training_error,iterations,grad_list

# def calculate_instances(train_samples,batch_size):
#
#     indices = list(range(train_samples.shape[0]))
#     np.random.shuffle(indices)
#     args = [iter(indices)] * batch_size
#     batch_indices = itertools.zip_longest(*args, fillvalue=None)
#     return [list(j for j in i if j is not None) for i in batch_indices]

# def logistic_regression_online(x_train,y,learning_rate,size_of_batch):
#
#     weights = [1,1,1]
#     batch_indices = calculate_instances(x_train,size_of_batch)
#     for  i in range(0,100000):
#         for batch in batch_indices:
#             output = x_train[batch].dot(weights)
#             predicted = 1 / (1 + np.exp(-output))
#             numerator = np.dot(x_train[batch].T, predicted - y[batch])
#             gradient = numerator / len(y[batch])
#             new_gradient = gradient * learning_rate
#             weights = weights - new_gradient
#
#         if gradient.mean() < 0.001:
#             break
#     print("Number of iteration = "+str(i))
#     return weights


mu1 =  [1, 0]
cov1 = [[1,0.75],[0.75,1]]
samples1 = generate_samples(mu1,cov1,500)
mu2 =  [0, 1.5]
cov2 = [[1,0.75],[0.75,1]]

samples2 = generate_samples(mu2,cov2,500)
training_data = np.concatenate((samples1,samples2))
label = np.array([0] * 500 + [1] * 500)

# add the bias
ones = np.ones((1000,1))
data = np.concatenate((ones,training_data), axis=1)
learning_rates = [1, 0.1, 0.01, 0.001]
for l in learning_rates:
    weights,error,iterations,gradients = logistic_regression_batch(data,label,l)

    test_samples_1 = generate_samples(mu1,cov1,500)
    test_samples_2 = generate_samples(mu2,cov2,500)
    Y_test = label
    X_test = np.vstack((test_samples_1,test_samples_2))
    X_test = np.concatenate((ones,X_test), axis=1)

    output_for_test = output = X_test.dot(weights)
    predicted_probability = 1 / (1 + np.exp(-output_for_test))
    predicted_label = []
    correct = 0
    i = 0
    total = 0

    for p in predicted_probability:
        if p < 0.5:
            predicted_label.append(0)
        elif p > 0.5:
            predicted_label.append(1)

    for p in predicted_label:
        if p == Y_test[i]:
            correct = correct + 1
        i = i + 1
        total = total + 1
    print("Accuracy metric when learning rate = "+str(l))
    print((correct/total)*100)
    display(test_samples_1, test_samples_2, weights, data,l)
    display_train_error(error,iterations,l)
    display_gradient_curve(gradients,iterations,l)


