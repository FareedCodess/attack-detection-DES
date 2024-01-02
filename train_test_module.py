import pandas as pd
import numpy as np
from numpy import argmax
from myKitsune import Kitsune
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix

"""
This file serves as a mini library to train, test, preprocess data, and calculate performance metrics.
"""


'''
 Separates a dataset into distinct subsets based on their labels.

Parameters:
- packets (pandas DataFrame): Dataset containing packets or instances.
- labels (pandas Series or DataFrame column): Labels associated with each packet.

Returns:
- tuple: Two distinct subsets of the dataset divided based on the provided labels,
where the first subset contains packets labeled as 0 and the second subset contains packets labeled as 1.
'''
def getDistinctDatasets(packets,labels):
    df=pd.concat([packets,labels],axis=1)
    groupDf = df.groupby('labels')
    return groupDf.get_group(0),groupDf.get_group(1)



'''
Trains the Kitsune model using the provided training dataset and saves the trained model to a file.

Parameters:
- trainingDataSet (numpy array): Training dataset used to train the Kitsune model.
- model (str): File path to save the trained Kitsune model.

Returns:
- list: Root Mean Squared Errors (RMSEs) calculated during the training process for each processed packet.
'''
def train(trainingDataSet,model):
    maxAE = 10          # maximum size for any autoencoder in the ensemble layer
    FMgrace = 5000      # the number of instances taken to learn the feature mapping (the ensemble's architecture)
    ADgrace = 50000     # the number of instances used to train the anomaly detector (ensemble itself)
    packets,features=trainingDataSet.shape
    K = Kitsune(features,maxAE,FMgrace,ADgrace)
    print("Running Kitsune:")
    RMSEs = []
    for i in range(packets):
        r=trainingDataSet[i]
        if i % 1000 == 0:
            print(i)
        rmse = K.proc_next_packet(r)
        RMSEs.append(rmse)
    pickle.dump(K, open(model, 'wb'))
    print("Model saved succesfully")
    return RMSEs
    

'''
Tests a trained model using a given test dataset.

Parameters:
- testDataSet (numpy array): Test dataset containing packets to be processed.
- model (str): File path to the trained model.

Returns:
- list: Root Mean Squared Errors (RMSEs) calculated for each packet processed by the trained model.
'''
def test(testDataSet,model):
    trainedModel = pickle.load(open(model, 'rb'))
    packets,features=testDataSet.shape
    print("Testing the packets:")
    RMSEs=[]

    for i in range(packets):
        if i%1000==0:
            print(i)
        
        r=testDataSet[i]
        rmse = trainedModel.proc_next_packet(r)
        RMSEs.append(rmse)
    return RMSEs


'''
Predicts binary outcomes based on Root Mean Squared Errors (RMSEs) and a specified threshold.

Parameters:
- RMSEs (array-like): List of Root Mean Squared Error values.
- threshold (float): Threshold value for classification.

Returns:
- list: Predicted binary outcomes where 1 represents values greater than or equal to the threshold,
and 0 represents values below the threshold.
'''
def predict(RMSEs,threshold):
    predicted=[]
    for i in RMSEs:
        if i>=threshold:
            predicted.append(1)
        else:
            predicted.append(0)
    return predicted


'''
Determines the best threshold value based on the Receiver Operating Characteristic (ROC) curve.

Parameters:
- tpr (array-like): True Positive Rate values.
- fpr (array-like): False Positive Rate values.
- thresholds (array-like): Threshold values used in the ROC curve calculation.

Returns:
- float: The threshold value that maximizes the difference (J) between True Positive Rate (TPR) and False Positive Rate (FPR).
'''
def bestThreshold(tpr,fpr,thresholds):
    J = tpr - fpr
    ix = argmax(J)
    return thresholds[ix]
    


"""
This function prints and plots the confusion matrix. Normalization can be applied by setting `normalize=True`.
"""
def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'


    cm = confusion_matrix(y_true, y_pred)
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# MIT License

# Copyright (c) 2023 Syed Fareed

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.
    
