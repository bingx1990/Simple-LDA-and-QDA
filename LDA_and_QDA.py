"""
    Implementation of LDA and QDA
"""
import numpy as np
import matplotlib.pyplot as plt
import scatter_contour as sc
import sample_dataset as sampledata
import diagonal_LDA as diag

#####    ==================   LDA and QDA   =====================  #####

"""
    get estimates of mus, sigmas and priors, for the covariance matrix,
    we use the MLE
"""
# @param - dataset: the generated features
# @param - numbs: the list of total numbers of each group
# @param - indicator: either "QDA", "LDA" or "diagonal LDA"
def estimates(dataset,numbs,indicator):
    k = int(dataset.shape[1])   # get the dimension of feature space
    count = 0                   # count how many data points we have passed
    for index,numb in enumerate(numbs):
        if index == 0:
            count = numb
            # get the number of "numb" data points
            data_group = dataset[0:numb]
            mu = np.mean(data_group,axis=0)
            mus_hat = mu
            pi_hat = np.array([numb])
            
            # estimate the covariance of each group for QDA
            if indicator == "QDA":
                var_hat = np.cov(np.transpose(data_group))*(numb-1)/numb
                var_hat = var_hat.reshape(1,k,k)
        else:
            # get the number of "numb" data points
            data_group = dataset[count:(count+numb)]
            count += numb           # update count
            mu = np.mean(data_group,axis=0)
            mus_hat = np.vstack([mus_hat,mu])
            pi_hat = np.append(pi_hat,numb)
            
            # estimate the covariance of each group for QDA
            if indicator == "QDA":
                var_tmp = np.cov(np.transpose(data_group))*(numb-1)/numb
                var_hat = np.concatenate((var_hat,var_tmp.reshape(1,k,k)))
                
    N = sum(pi_hat)              # total number of observations
    pi_hat = pi_hat/float(N)     # priors
    
    # estimate the covariance for LDA
    if indicator == "LDA":
        var_hat = np.cov(np.transpose(dataset))*(N-1)/N
        var_hat = var_hat.reshape(1,k,k)
    
    return([mus_hat, pi_hat, var_hat])
# @return - mus_hat: the means of different groups listed in a k by p array
# @return - pi_hat: the list of the k priors
# @return - var_hat: the array of estiamted covariance matrices

### classification for test sets
def classification(testset,mus_hat,pi_hat,var_hat):
    k = int(testset.shape[1])
    labels = np.array([])
    # for the case of using LDA 
    if var_hat.shape[0] == 1: 
        for point in testset:
            probs_log = np.array([])
            for index,pi in enumerate(pi_hat):
                mu = mus_hat[index].reshape(1,k)
                sigma_inv = np.linalg.inv(var_hat)
                tmp = np.log(pi)-0.5*mu.dot(sigma_inv).dot(np.transpose(mu))+mu.dot(sigma_inv).dot(point.reshape(k,1))
                probs_log = np.append(probs_log,tmp)
            index = np.argmax(probs_log)
            labels = np.append(labels,index)
    else:     # QDA
        for point in testset:
            probs_log = np.array([])
            for index,pi in enumerate(pi_hat):
                mu = mus_hat[index].reshape(1,k)
                sigma = var_hat[index]
                sigma_inv = np.linalg.inv(sigma)
                tmp = np.log(pi)-0.5*mu.dot(sigma_inv).dot(np.transpose(mu))+mu.dot(sigma_inv).dot(point.reshape(k,1))\
                      -0.5*point.dot(sigma_inv).dot(point.reshape(k,1))
                probs_log = np.append(probs_log,tmp)
            index = np.argmax(probs_log)
            labels = np.append(labels,index)
    return labels

if __name__ == "__main__":
    """
    mus = np.array([[-4,1,-1],[-4,-5,5],[5,0,5]])
    Sigma = np.array([[[2,0,0],[0,2,0],[0,0,2]],
                      [[4,1,1],[1,3,1],[1,1,2]],
                      [[3,0,1],[0,2,0],[1,0,2]]])
    """
    mus = np.array([[4,-2],[-5,-2],[0,5]])
    Sigma = np.array([[[2,1],[1,2]],
                      [[2,1],[1,2]],
                      [[2,1],[1,2]]])
    training_numbs = [3000,3000,3000]
    test_numbs = [100,100,100]
    training_data = sampledata.generate_dataset(training_numbs,mus,Sigma)
    test_data = sampledata.generate_dataset(test_numbs,mus,Sigma)
    training_features = training_data[0]
    test_features = test_data[0]
    training_labels = training_data[1]
    test_labels = test_data[1]
    ests = estimates(training_features,training_numbs,"LDA")
    
    # diagnal LDA
    #ests = diag.estimates(training_features,training_numbs)
    
    training_preds = classification(training_features,ests[0],ests[1],ests[2])
    test_preds = classification(test_features,ests[0],ests[1],ests[2])
    print "The mis-classifiation error for the training set is {}".format(np.mean(training_preds!=training_labels))
    print "The mis-classifiation error for the test set is {}".format(np.mean(test_preds!=test_labels))

    # make plots
    sc.scatterplot(training_features,training_numbs)
    sc.contour(-10,10,0.1,ests)
    plt.show()

