"""
    This script generates new data in the form of a list which contains
    two arrays. One is the labels, the other one is the N by D array of
    corresponding features.
    eg: [[x11,x21,...,xD1],[x21,x22,...,x2D]...[xN1,xN2,...,xND]]
"""
import numpy as np

# @param - numbs: the list of numbers for each group. eg: [123,232,1231]
# @param - mus: the G by D array of means.
#               eg:[[mu11,mu12,...,mu1D],[muG1,muG2,...,muGD]...]
# @param - Sigma: the array of covariance matrix.
def generate_dataset(numbs, mus, Sigma):
    labels = np.array([])
    for i,n in enumerate(numbs):
        labels = np.append(labels,np.linspace(i,i,n))
        if i == 0:
            sample = np.random.multivariate_normal(mus[i], Sigma[i], n)
            data = sample
        else:
            sample = np.random.multivariate_normal(mus[i], Sigma[i], n)
            data = np.vstack((data,sample))
    return [data,labels]
# @return - data: the features arranged in the form of a N by p array
# @return - labels: the corresponding labels as an array of length N

