"""
    This script aims to draw the 2-d scatter plot of points
    in different groups and the decision boundary
"""
import numpy as np
import matplotlib.pyplot as plt
import classification as cl

# draw scatter plots
def scatterplot(training_features, training_numbs):
    colors = ["g","r","y","p","b","m"]
    count = 0
    for index,numb in enumerate(training_numbs):
        if index == 0:
            count = numb
            training_x1 = training_features[:numb,0]
            training_x2 = training_features[:numb,1]
            plt.scatter(training_x1,training_x2,c=colors[index],marker="o",s=15)
        else:
            training_x1 = training_features[count:(numb+count),0]
            training_x2 = training_features[count:(numb+count),1]
            count += numb
            plt.scatter(training_x1,training_x2,c=colors[index],marker="o",s=15)

# draw the decision boundary
def contour(lower,upper,interval,ests,diag=False):
    x = np.arange(lower,upper,interval)
    y = np.arange(lower,upper,interval)
    xx,yy = np.meshgrid(x,y)
    for i,line in enumerate(xx):
        tmp = np.array([])
        for j,coord in enumerate(line):
            point = np.array([coord,yy[i,j]]).reshape(1,2)
            if not diag:
                pred = cl.classification(point,ests[0],ests[1],ests[2])
            else:
                pred = cl.diag_classification(point,ests[0],ests[1],ests[2])
            tmp = np.append(tmp,pred)
        if i==0:
            z = tmp
        else:
            z = np.vstack((z,tmp))
    plt.contour(x,y,z)
