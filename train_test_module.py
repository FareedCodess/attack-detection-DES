import pandas as pd
import numpy as np
from numpy import sqrt, argmax
from myKitsune import Kitsune
import time
import pickle
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix


def getDistinctDatasets(packets,labels):
    df=pd.concat([packets,labels],axis=1)
    groupDf = df.groupby('labels')
    return groupDf.get_group(0),groupDf.get_group(1)


def train(trainingDataSet,model):
    maxAE = 10 #maximum size for any autoencoder in the ensemble layer
    FMgrace = 5000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
    ADgrace = 50000 #the number of instances used to train the anomaly detector (ensemble itself)
    packets,features=trainingDataSet.shape
    # Build Kitsune
    K = Kitsune(features,maxAE,FMgrace,ADgrace)
    print("Running Kitsune:")
    RMSEs = []
    start = time.time()

    for i in range(packets):
        r=trainingDataSet[i]
        if i % 1000 == 0:
            print(i)
        rmse = K.proc_next_packet(r)
        

        RMSEs.append(rmse)

    
    
    pickle.dump(K, open(model, 'wb'))
    print("Model saved succesfully")

    return RMSEs
    

    
def test(testDataSet,model):
    trainedModel = pickle.load(open(model, 'rb'))
    packets,features=testDataSet.shape
    print("Testing the packets:")
    RMSEs=[]

    for i in range(packets):
        if i%1000==0:
            print(i)
        
        r=testDataSet[i]
        #start timer
        rmse = trainedModel.proc_next_packet(r)
        #endtimer
        RMSEs.append(rmse)
    return RMSEs

def predict(RMSEs,threshold):
    
    predicted=[]
    
    for i in RMSEs:
        if i>=threshold:
            predicted.append(1)

        else:
            predicted.append(0)
        
    
    return predicted

def bestThreshold(tpr,fpr,thresholds):
    # gmeans = sqrt(tpr * (1-fpr))
    # # locate the index of the largest g-mean
    # ix = argmax(gmeans)
    # print('Best Threshold=%f, G-Mean=%.3f' % (thresholds[ix], gmeans[ix]))
    # get the best threshold using Youden's J statistic
    J = tpr - fpr
    ix = argmax(J)

    return thresholds[ix]
    

def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=False,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
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
    
