"""
    This script uses LDA and QDA to classify the test point
    including the Reduced Rank LDA(RLDA) and diagonal DA
"""
import numpy as np

### eigenvalue decomposition of a list of covariance matrices
### and the determinant, eg: Sigma = U'DU
# @param var: the array of the estimated covariance matrices
def eigen_decomp(var):
    if len(var) == 1:
        eigens,Us = np.linalg.eigh(var[0])
        log_dets = sum(np.log(eigens))
    else:
        eigens = []
        Us = []
        log_dets = []
        for item in var:
            eigen,U = np.linalg.eigh(item)
            log_dets = np.append(log_dets,sum(np.log(eigen)))
            Us.append(U)
            eigens.append(eigen)
    return [eigens,Us,log_dets]
# @return eigen_decomps: the list of eigenvalues, corresponding right
#         eigenvectors and the logarithm determinant
#         eg: [[array(2,3)],[array([[1,0],[0,1]]]]

### sphere the data by using eigen decomponents
# @param data: k by p 
# @param U: RHS eigenvectors
# @param eigen: array containing eigenvalues
def sphere(data, U, eigen):
    spherical_data = data.dot(U.T).dot(np.diag(eigen**(-0.5)))  # (k,p)
    return spherical_data
# @return spherical_data: k by p spherical version

### calculate the reduced subspace and project data onto it
# @param data: spherical data, k by p
# @param mus: spherical centroids, p by p
# @param r: rank to which the feature space is reduced
def reduce_rank(data, mus, r, p):
    # calculate the between-group covariance matrix
    B_hat = np.cov(mus,rowvar=False).reshape(1,p,p)   # (p,p)
    eig_B,U_B = eigen_decomp(B_hat)[:2] 
    U_B_res = U_B[:,-r:]        # (p,r)
    data = data.dot(U_B_res)    # (n,r)
    mus = mus.dot(U_B_res)      # (k,r)
    return [data,mus]
   

### classification of the test points according to different methods
### k denotes the number of labeled groups, p denotes the dim of features
# @param testset: n by p
# @param mus_hat: k by p
# @param pi_hat: a length k list of priors
# @param var_hat: a 3d array which contains the estimated covariance matrices
def classification(testset, mus_hat, pi_hat, var_hat, r = False):
    p = var_hat.shape[1]         # dimension of feature space
    labels = np.array([])
    if var_hat.shape[0] != 1:    # QDA
        eigs,Us,log_dets = eigen_decomp(var_hat)
        # sphere the centroids according to different covariance matrices
        for index,pi in enumerate(pi_hat):
            mu_tmp = mus_hat[index,:].reshape(1,p)
            if index == 0:
                mus_tran = sphere(mu_tmp,Us[index],eigs[index])  #(1,p)
            else:
                mus_tran = np.vstack((mus_tran,sphere(mu_tmp,Us[index],eigs[index])))
        # classify testset
        for point in testset:
            point = point.reshape([1,p])
            probs_log = np.array([])
            for index,pi in enumerate(pi_hat):
                # sphere the test point
                point_tran = sphere(point,Us[index],eigs[index])   # (1,p)
                mu_tran = mus_tran[index,:].reshape(1,p)
                product = np.dot(point_tran-mu_tran,(point_tran-mu_tran).T)
                tmp = np.log(pi)-0.5*product-0.5*log_dets[index]
                probs_log = np.append(probs_log,tmp)
            index = np.argmax(probs_log)
            labels = np.append(labels,index)
    else:
        # get the eigen-decomp of the estimated covariance matrix(within)
        eig,U = eigen_decomp(var_hat)[:2]
        # sphere the centroids and the test set
        mus_tran = sphere(mus_hat,U,eig)    # (k,p)
        test_tran = sphere(testset,U,eig)   # (n,p)
        
        if not r:             # LDA
            for point in test_tran:
                probs_log = np.array([])
                for index,pi in enumerate(pi_hat):
                    mu = mus_tran[index,:].reshape(1,p)
                    product = np.log(pi)-0.5*(point-mu).dot((point-mu).T)
                    probs_log = np.append(probs_log,product)
                index = np.argmax(probs_log)
                labels = np.append(labels,index)
        else:       # RLDA with reduced rank r
            test_tran, mus_tran = reduce_rank(test_tran,mus_tran,r,p)
            for point in test_tran:
                probs_log = np.array([])
                for index,pi in enumerate(pi_hat):
                    mu = mus_tran[index,:].reshape(1,r)
                    tmp = np.log(pi)-0.5*(point-mu).dot(np.transpose(point-mu))
                    probs_log = np.append(probs_log,tmp)
                index = np.argmax(probs_log)
                labels = np.append(labels,index)
                return [labels,test_tran]
    return labels
            
### classification by using diagonal Discriminant Analysis
def diag_classification(testset,mus_hat,pi_hat,var_hat):
    p = int(testset.shape[1])
    labels = np.array([])
    # for the case of using LDA 
    if var_hat.shape[0] == 1:
        sigma_inv = np.diag(np.diagonal(var_hat[0])**(-1))
        for point in testset:
            point = point.reshape(1,p)
            probs_log = np.array([])
            for index,pi in enumerate(pi_hat):
                mu = mus_hat[index].reshape(1,p)
                tmp = np.log(pi)-0.5*mu.dot(sigma_inv).dot(mu.T)+\
                      mu.dot(sigma_inv).dot(point.T)
                probs_log = np.append(probs_log,tmp)
            index = np.argmax(probs_log)
            labels = np.append(labels,index)
    else:     # QDA
        for point in testset:
            point = point.reshape(1,p)
            probs_log = np.array([])
            for index,pi in enumerate(pi_hat):
                mu = mus_hat[index].reshape(1,p)
                sigma = np.diag(np.diagonal(var_hat[index])**(-1))
                tmp = np.log(pi)-0.5*(point-mu).dot(sigma_inv).dot((point-mu).T)\
                      -0.5*np.log(np.linalg.det(sigma))
                probs_log = np.append(probs_log,tmp)
            index = np.argmax(probs_log)
            labels = np.append(labels,index)
    return labels

            
                
            
        
