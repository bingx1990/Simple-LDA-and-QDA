"""
    Diagonal LDA
"""
import numpy as np

# @param - dataset: the generated features, N by D array
# @param - numbs: the list of total numbers of each group
def estimates(dataset, numbs):
    k = int(dataset.shape[1])   # get the dimension of feature space
    count = 0                   # count how many data points we have passed
    var_hat = np.zeros([k])
    for index,numb in enumerate(numbs):
        if index == 0:
            count = numb
            # get the number of "numb" data points
            data_group = dataset[0:numb]
            mu = np.mean(data_group,axis=0)
            mus_hat = mu
            pi_hat = np.array([numb])
            # estimate the diagonal entries of the covariance matrix
            var_hat += np.sum((data_group-mu)**2,axis=0)
        else:
            # get the number of "numb" data points
            data_group = dataset[count:(count+numb)]
            count += numb           # update count
            mu = np.mean(data_group,axis=0)
            mus_hat = np.vstack([mus_hat,mu])
            pi_hat = np.append(pi_hat,numb)
            var_hat += np.sum((data_group-mu)**2,axis=0)
    N = sum(pi_hat)              # total number of observations
    pi_hat = pi_hat/float(N)     # priors

    var_hat = var_hat/float(N-len(numbs))
    var_hat = np.diag(var_hat)
    return([mus_hat, pi_hat, var_hat])
# @return - mus_hat: the means of different groups listed in a k by p array
# @return - pi_hat: the list of the k priors
# @return - var_hat: the diagonal array of estiamted covariance matrices


