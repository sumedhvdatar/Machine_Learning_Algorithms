#Author : Sumedh Vilas Datar

import numpy as np
import math
from matplotlib import pyplot as plt
from scipy.stats import multivariate_normal
from sklearn.metrics import confusion_matrix


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


def calculate_gaussain_probabaility(mean,std,x):
    p = multivariate_normal.pdf(x, mean, std)
    return p
# Importing necessary libraries...
import collections
import numpy as np
def calculate_prior_probability(labels):
    y_dict = collections.Counter(labels)
    pre_probab = np.ones(2)
    for i in range(0, 2):
        pre_probab[i] = y_dict[i] / len(labels)
    return pre_probab


def myNB(X,y,samples_test,labels_test,separated_dict):
    pred = []
    posterior = []
    error = []
    i = 0
    for test_data in samples_test:
        i = i + 1
        calculate_mean_c1 = np.mean(separated_dict[0],axis=0)
        calculate_mean_c2 = np.mean(separated_dict[1],axis=0)

        calculate_std_c1 = np.std(separated_dict[0],axis=0)
        calculate_std_c2 = np.std(separated_dict[1],axis=0)

        gaussian_prob_c1 = calculate_gaussain_probabaility(calculate_mean_c1, calculate_std_c1, test_data)
        gaussian_prob_c2 = calculate_gaussain_probabaility(calculate_mean_c2, calculate_std_c2, test_data)

        prior_prob = calculate_prior_probability(y)

        probability_c0_given_test_data = gaussian_prob_c1 * prior_prob[0]
        probability_c1_given_test_data = gaussian_prob_c2 * prior_prob[1]
        prediction_list = [probability_c0_given_test_data,probability_c1_given_test_data]
        predicted_class = np.asarray(prediction_list).argmax()
        prediction = max(prediction_list)
        pred.append(predicted_class)
        posterior.append(prediction)
        error.append(1-prediction)
    return pred,posterior,error

def calculate_accuracy(predicted_labels,ground_truth_label):
    total = len(ground_truth_label)
    count = 0
    i = 0
    for label in predicted_labels:
        if label == ground_truth_label[i]:
            count = count + 1
        i = i + 1
    accuracy = (count / total) * 100
    return accuracy




mu1 =  [1, 0]
cov1 = [[1,0.75],[0.75,1]]
samples1 = generate_samples(mu1,cov1,500)
mu2 =  [0, 1]
cov2 = [[1,0.75],[0.75,1]]
samples2 = generate_samples(mu2,cov2,500)

labels_1 = [0]*500
labels_2 = [1]*500

X= samples1 + samples2
Y = labels_1+labels_2


test_samples_1 = generate_samples(mu1,cov1,500)
test_samples_2 = generate_samples(mu2,cov2,500)
Y_test = labels_1 + labels_2
X_test = np.vstack((test_samples_1,test_samples_2))


separated = {}
separated[0] = samples1
separated[1] = samples2
pred, posterior, err = myNB(X,Y,X_test,Y_test,separated)

#END OF QUESTION 1#######################################

#Questino 2

acc = calculate_accuracy(pred,Y_test)
print("Accuracy = "+str(acc))

tn, fp, fn, tp = confusion_matrix(Y_test,pred).ravel()
# print(tn,fp,fn,tp)
tpr = tp/ (tp+fn)
print("True Positive Rate "+str(tpr))
print("Recall "+str(tpr))
fpr = fp/(fp+tn)
print("False Positive Rate "+str(fpr))
precision = tp/(tp+fp)
print("Precision = "+str(precision))
indices = [i for i in range(500)]
fig, ax = plt.subplots()
ax.scatter(test_samples_1[:,0],test_samples_1[:,1],color = "black")
ax.scatter(test_samples_2[:,0],test_samples_1[:,1],color = "red")
plt.title('1000 samples from a 2D Gaussian distribution with 2 classes')
plt.ylabel('x2')
plt.xlabel('x1')
plt.show()

##END OF QUESTION 2#####################################



