"""
    Diagonal LDA and Regularized LDA
"""
import numpy as np

# @param - dataset: the generated features, N by D array
# @param - numbs: the list of total numbers of each group
def diag_estimates(dataset, numbs):
    p = int(dataset.shape[1])   # get the dimension of feature space
    count = 0                   # count how many data points we have passed
    var_hat = np.zeros([p])
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
    var_hat = np.diag(var_hat).reshape(1,p,p)
    return([mus_hat, pi_hat, var_hat])
# @return - mus_hat: the means of different groups listed in a k by p array
# @return - pi_hat: the list of the k priors
# @return - var_hat: the diagonal array of estiamted covariance matrices

"""
    get estimates of mus, sigmas and priors, for the covariance matrix,
    we use the regularized version which is
    Sigma_k = alpha*Sigma_k + (1-alpha)*Sigma_hat
"""
# @param - dataset: the generated features
# @param - numbs: the list of total numbers of each group
# @param - alpha
def reg_estimates(dataset,numbs,alpha):
    p = int(dataset.shape[1])   # get the dimension of feature space
    k = len(numbs)              # get the number of groups
    count = 0                   # count how many data points we have passed
    for index,numb in enumerate(numbs):
        if index == 0:
            count = numb
            # get the number of "numb" data points
            data_group = dataset[0:numb]
            mu = np.mean(data_group,axis=0)
            mus_hat = mu
            pi_hat = np.array([numb])
            var_pool = np.cov(data_group,rowvar=False)*(numb-1)
            
            # estimate the covariance of each group for QDA
            var_QDA = var_pool/numb
            var_QDA = var_QDA.reshape(1,p,p)
        else:
            # get the number of "numb" data points
            data_group = dataset[count:(count+numb)]
            count += numb           # update count
            mu = np.mean(data_group,axis=0)
            mus_hat = np.vstack([mus_hat,mu])
            pi_hat = np.append(pi_hat,numb)
            var_group = np.cov(data_group,rowvar=False)*(numb-1)
            
            # estimate the covariance of each group for QDA
            var_QDA_tmp = var_group/numb
            var_QDA = np.concatenate((var_QDA,var_QDA_tmp.reshape(1,p,p)))
            # pool the variance together
            var_pool += var_group
                
    N = sum(pi_hat)              # total number of observations
    pi_hat = pi_hat/float(N)     # priors
    
    # estimate the covariance for LDA
    var_LDA = (var_pool/float(N-k)).reshape(1,p,p)

    # calculate the regularized covariance
    for index,group in enumerate(numbs):
        var_reg_tmp = alpha*var_QDA[index]+(1-alpha)*var_LDA[0]
        var_reg_tmp = var_reg_tmp.reshape(1,p,p)
        if index==0:
            var_hat = var_reg_tmp
        else:
            var_hat = np.concatenate((var_hat,var_reg_tmp))
    return([mus_hat, pi_hat, var_hat])
# @return - mus_hat: the means of different groups listed in a k by p array
# @return - pi_hat: the list of the k priors
# @return - var_hat: the array of estiamted covariance matrices

"""
    get estimates of mus, sigmas and priors, for the covariance matrix,
    we use the MLE
"""
# @param - dataset: the generated features
# @param - numbs: the list of total numbers of each group
# @param - indicator: either "QDA" or "LDA"
def estimates(dataset,numbs,indicator):
    p = int(dataset.shape[1])   # get the dimension of feature space
    k = len(numbs)              # get the number of groups
    count = 0                   # count how many data points we have passed
    for index,numb in enumerate(numbs):
        if index == 0:
            count = numb
            # get the number of "numb" data points
            data_group = dataset[0:numb]
            mu = np.mean(data_group,axis=0)
            mus_hat = mu
            pi_hat = np.array([numb])
            var_pool = np.cov(data_group,rowvar=False)*(numb-1)
            
            # estimate the covariance of each group for QDA
            if indicator == "QDA":
                var_hat = var_pool/numb
                var_hat = var_hat.reshape(1,p,p)
        else:
            # get the number of "numb" data points
            data_group = dataset[count:(count+numb)]
            count += numb           # update count
            mu = np.mean(data_group,axis=0)
            mus_hat = np.vstack([mus_hat,mu])
            pi_hat = np.append(pi_hat,numb)
            var_group = np.cov(data_group,rowvar=False)*(numb-1)
            
            # estimate the covariance of each group for QDA
            if indicator == "QDA":
                var_tmp = var_group/numb
                var_hat = np.concatenate((var_hat,var_tmp.reshape(1,p,p)))
            else:
                var_pool += var_group
                
    N = sum(pi_hat)              # total number of observations
    pi_hat = pi_hat/float(N)     # priors
    
    # estimate the covariance for LDA
    if indicator == "LDA":
        var_hat = (var_pool/float(N-k)).reshape(1,p,p)

    return([mus_hat, pi_hat, var_hat])
# @return - mus_hat: the means of different groups listed in a k by p array
# @return - pi_hat: the list of the k priors
# @return - var_hat: the array of estiamted covariance matrices
