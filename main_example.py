"""
    Implementation of LDA and QDA
"""
import numpy as np
import matplotlib.pyplot as plt
import scatter_contour as sc
import sample_dataset as sampledata
import estimate
import classification as cl


if __name__ == "__main__":
    
    mus = np.array([[-4,-2],[0,4],[5,1]])
    Sigma = np.array([[[4,-1],[-1,2]],
                      [[3,1.5],[1.5,3]],
                      [[2,0],[0,4]]])
    
    training_numbs = [2000,2000,2000]
    test_numbs = [100,100,100]
    training_data = sampledata.generate_dataset(training_numbs,mus,Sigma)
    test_data = sampledata.generate_dataset(test_numbs,mus,Sigma)
    training_features = training_data[0]
    test_features = test_data[0]
    training_labels = training_data[1]
    test_labels = test_data[1]
    
    #ests = estimate.estimates(training_features,training_numbs,"LDA")

    #regularized
    ests = estimate.reg_estimates(training_features,training_numbs,0.9)

    training_preds = cl.classification(training_features,ests[0],ests[1],ests[2])
    test_preds = cl.classification(test_features,ests[0],ests[1],ests[2])
    
    # diagnal LDA
    #ests = estimate.diag_estimates(training_features,training_numbs)
    #training_preds = cl.diag_classification(training_features,ests[0],ests[1],ests[2])
    #test_preds = cl.diag_classification(test_features,ests[0],ests[1],ests[2])
    

    """
    # reduced rank LDA
    training_preds,training_tran = cl.classification(training_features,ests[0],ests[1],ests[2],r=2)
    test_preds,test_tran = cl.classification(test_features,ests[0],ests[1],ests[2],r=2)
    sc.scatterplot(training_tran,training_numbs)
    plt.show()
    """
    
    print "The mis-classifiation error for the training set is {}".format(np.mean(training_preds!=training_labels))
    print "The mis-classifiation error for the test set is {}".format(np.mean(test_preds!=test_labels))

    # make plots
    sc.scatterplot(training_features,training_numbs)
    sc.contour(-10,10,0.1,ests,diag=False)
    plt.show()

