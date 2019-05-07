#Author : Sumedh Vilas Datar
#This file contains the implementation details of KMeans clustering.
import numpy as np
from matplotlib import pyplot as plt


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

def euclidena_distance(a,b):
    """
    This function calculates the euclidena distance
    :param a: Coordinates
    :param b: Coordinates
    :return: distance value
    """
    dist = np.linalg.norm(a - b)
    return dist

def create_clusters(X,centroids,cluster_index):
    """
    This function creates clusters and indexes the entire sample dataset
    :param X:
    :param centroids:
    :param cluster_index:
    :return: a new array with indexed values where each index will have 1 cluster.
    """
    for i in range(0, X.shape[0]):
        min_distance = 1000
        centroid_index = -1
        for j in range(0, k):
            distance = euclidena_distance(centroids[j], X[i])
            if distance < min_distance:
                min_distance = distance
                centroid_index = j
        cluster_index[i] = centroid_index
    return cluster_index

def new_centroids(X,index_array,cluster_size):
    """
    This method averages across all and gives the output for new centroid.
    :param X:
    :param index_array:
    :param cluster_size:
    :return: new centroid values
    """
    new_centroids = []
    for i in range(0,cluster_size):
        sum = np.zeros((1,2))
        count = 0
        for j in range(0,len(index_array)):
            if i == index_array[j]:
                sum = sum + X[j]
                count = count + 1
        avg = sum/count
        avg = avg.tolist()
        new_centroids.append(avg[0])
    return new_centroids



def mykmeans(X,centroids,k):
    iteration = 1
    i = 0
    ref_centroids = np.zeros((k,2))
    norm = euclidena_distance(centroids,ref_centroids)
    while(norm > 0.001):
        if i == 10000:
            break
        cluster_index = [0]*X.shape[0]
        final_cluster_index_array = create_clusters(X,centroids,cluster_index)
        updated_centroids = new_centroids(X,final_cluster_index_array,k)
        norm = euclidena_distance(np.asarray(updated_centroids),np.asarray(centroids))
        centroids = updated_centroids
        # print(i)
        i = i + 1
    print("It took "+str(i)+" iteration to train")
    return centroids
#########################################################################################
#Chnage mu parameters here if required
mu1 =  [1, 0]
cov1 = [[0.9,0.4],[0.4,0.9]]
samples1 = generate_samples(mu1,cov1,500)

mu2 =  [0,1.5]
cov1 = [[0.9,0.4],[0.4,0.9]]
samples2 = generate_samples(mu2,cov1,500)
samples = np.concatenate((samples1,samples2),axis=0)


#Set the value to either 2 or 4.
#2 : will give two clusters
# 4: will give four clusters.
k = 4
#############################################################################################
if k == 4:
    #If required change the centers here
    centers = [[10, 10], [-10, -10], [10, -10], [-10, 10]]
else:
    #here as well , if required..
    centers = [[10,10],[-10,-10]]
final_centroid = np.asarray(mykmeans(samples,centers,k))



# clusters = np.zeros(len(X))
#Plot the data
fig, ax = plt.subplots()
ax.scatter(samples1[:,0],samples1[:,1],color = "blue")
ax.scatter(samples2[:,0],samples2[:,1],color="blue")
ax.scatter(final_centroid[:,0],final_centroid[:,1],color="black")
plt.title('1000 samples from a 2D Gaussian distribution')
plt.ylabel('x2')
plt.xlabel('x1')
plt.show()

#plot for two
if k == 2:
    cluster_0 = []
    cluster_1 = []

    for s in samples:
        label_index = 0
        cluster_label = -1
        min = 1000
        for centroid in final_centroid:
            dist = euclidena_distance(centroid,s)
            if dist <= min:
                cluster_label = label_index
                min = dist
            label_index = label_index + 1
        eval("cluster_" + str(cluster_label)).append(s)
    cluster_0 = np.asarray(cluster_0)
    cluster_1 = np.asarray(cluster_1)
    # Plot when you have two clusters
    fig, ax = plt.subplots()
    # cluster_0 = [cluster_0]
    ax.scatter(cluster_0[:, 0], cluster_0[:, 1], color="yellow")
    ax.scatter(cluster_1[:, 0], cluster_1[:, 1], color="red")
    ax.scatter(final_centroid[:, 0], final_centroid[:, 1], color="black")
    plt.title('1000 samples from a 2D Gaussian distribution')
    plt.ylabel('x2')
    plt.xlabel('x1')
    # plt.show()
    plt.show()
    print("New centroids for K = 2 are :- ")
    print(final_centroid)

#plot for 4
elif k == 4:
    cluster_0 = []
    cluster_1 = []
    cluster_2 = []
    cluster_3 = []

    for s in samples:
        label_index = 0
        cluster_label = -1
        min = 1000
        for centroid in final_centroid:
            dist = euclidena_distance(centroid, s)
            if dist <= min:
                cluster_label = label_index
                min = dist
            label_index = label_index + 1
        eval("cluster_" + str(cluster_label)).append(s)
    # print(samples)
    cluster_0 = np.asarray(cluster_0)
    cluster_1 = np.asarray(cluster_1)
    cluster_2 = np.asarray(cluster_2)
    cluster_3 = np.asarray(cluster_3)
    #Plot when you have 4 clusters
    fig, ax = plt.subplots()
    # cluster_0 = [cluster_0]
    ax.scatter(cluster_0[:, 0], cluster_0[:, 1], color="yellow")
    ax.scatter(cluster_1[:, 0], cluster_1[:, 1], color="red")
    ax.scatter(cluster_2[:, 0], cluster_2[:, 1], color="green")
    ax.scatter(cluster_3[:, 0], cluster_3[:, 1], color="orange")
    ax.scatter(final_centroid[:, 0], final_centroid[:, 1], color="black")
    plt.title('1000 samples from a 2D Gaussian distribution')
    plt.ylabel('x2')
    plt.xlabel('x1')
    plt.show()
    print("New centroids for K = 4 are :- ")
    print(final_centroid)




