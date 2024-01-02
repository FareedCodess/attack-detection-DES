from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from myKitsune import Kitsune
import numpy as np
import pandas as pd
from train_test_module import getDistinctDatasets,bestThreshold,plot_confusion_matrix,predict
import matplotlib.pyplot as plt

'''
This file serves the following purposes:
1. Trains the model using benign data from a specified dataset.
2. Tests the trained model on the training set under three conditions:
    a. Full ensemble layer activation
    b. Single AE (Autoencoder) activation (Only one AE active per testing round)
    c. Multiple active AEs based on a specified filter size and most accurate AEs based on RMSE.
3. Generates Visualizations:
    - ROC Curve
    - Confusion Matrix
    - Lineplot of RMSEs (Root Mean Squared Errors) to assess model performance under different conditions.
'''

# Data Loading and Preprocessing
# -------------------------------------------------------------------------------------------------------------------------------------
# Code related to loading datasets, cleaning, and preprocessing steps

# Legend for variables:
# L=labels ; B= benign ; M= malicious 

# loading packet features and its labels
print("Loading Data ...")
packetsDf=pd.read_csv("C:/KFUPM/Research/Term221_Kitsune/Kitsune Dataset from Kaggle/Video Injection/Video_Injection_dataset.csv",names=list(range(1,116)))
labelDf=pd.read_csv("C:/KFUPM/Research/Term221_Kitsune/Kitsune Dataset from Kaggle/Video Injection/Video_Injection_labels.csv",names=['x','labels'])
labelDf=labelDf.drop(0)
labelDf.drop(columns='x',inplace=True)
labelDf=labelDf.astype(np.int64)
labelDf=labelDf.reset_index(drop=True)

# filtering benign and malicious samples
benignDf,maliciousDf=getDistinctDatasets(packetsDf,labelDf['labels'])

# splitting benign dataset int two folds of test size 0.4
B_train, B_test, L_B_train, L_B_test = train_test_split(benignDf[list(range(1,116))].to_numpy(),benignDf['labels'].to_numpy(), 
                                                        test_size=0.8 ,shuffle= False,random_state=42)

# This parameter determine the number of training packets on which the model will execute.
trainingPacketsCount = 10000

L_B_train   = L_B_train[0:trainingPacketsCount]
B_train     = B_train[0:trainingPacketsCount]   
packets,features=B_train.shape


# Model Training
# -------------------------------------------------------------------------------------------------------------------------------------
# Code for defining and training model


# KitNET params:
maxAE = 11          # maximum size for any autoencoder in the ensemble layer
FMgrace = None      # the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = trainingPacketsCount    # the number of instances used to train the anomaly detector (ensemble itself)

customfeatureMap = [[(i*5 + j) for j in range(5)]for i in range(int(features/5))]
K = Kitsune(features,maxAE,FMgrace,ADgrace,feature_map=customfeatureMap)
print("Training phase :")
for z in range(trainingPacketsCount):
    r = B_train[z]
    K.proc_next_packet(r)
    if z % 10000 == 0:
        print(z)


# Evaluation and Visualization
# -------------------------------------------------------------------------------------------------------------------------------------
# Code for evaluating model and generating visualizations

fullEnsembleRMSEs = []
print("Collecting RMSEs for Full Ensemble by testing on the training dataset : ")
for z in range(trainingPacketsCount):
    r = B_train[z]
    fullEnsembleRMSEs.append(K.proc_next_packet(r))
    if z % 10000 == 0:
        print(z)

RMSEsSingleActivation = []
print("Collecting RMSEs by Testing on training dataset by activating one autoencoder.") 
for i in range(len(K.AnomDetector.ensembleLayer)):
    RMSEsOneAE =[]
    print (f"Current Active AE : {i}")
    for j in range(trainingPacketsCount):
        r  =  B_train[j]
        RMSE = K.proc_next_packet(r,i)  # i -> index of the active autencoder
        RMSEsOneAE.append(RMSE)
        if j % 10000 == 0:
            print(j)
    RMSEsSingleActivation.append(RMSEsOneAE)


"""
This code numbers the AEs based on the mean RMSEs produced by the Single Activation of each AE.
"""
MeanRMSEs = []
for i in RMSEsSingleActivation:
    arr = np.array(i)
    filtered_arr = arr[arr <= 1]                # Remove elements greater than 1 (Outliers)
    mean_of_filtered = np.mean(filtered_arr)    # Calculate the mean of the remaining elements
    MeanRMSEs.append(mean_of_filtered)

indexed_rmse = list(enumerate(MeanRMSEs))
sorted_rmse = sorted(indexed_rmse, key=lambda x: x[1],reverse=False)
sorted_values = [rmse for _, rmse in sorted_rmse]
original_indexes = [index for index, _ in sorted_rmse]

# Inserting full ensemble RMSE for model performance comparison.
FE_arr = np.array(fullEnsembleRMSEs)
FE_filtered_arr = FE_arr[FE_arr <= 1]           # Remove elements greater than 1 (Outliers)
FE_mean_of_RMSE = np.mean(FE_filtered_arr)
sorted_rmse.insert(0,('FE',FE_mean_of_RMSE))



# Note : Adjust the filter percentage in order to determine the number of active AEs.
filter = 0.5      # activating 50% autoencoders       
selected_indexes = [original_indexes[i] for i in range(int(len(original_indexes)*filter))]
print("Selected Autoencoders: ", selected_indexes)
RMSEsSelectedAE =[]
print("Testing on training data by activating selected autoencoders based on the RMSEs of individual AEs :") 
for i in range(trainingPacketsCount):
    if i % 10000 == 0:
        print(i)
    r = B_train[i]
    RMSE = K.proc_next_packet(r,selected_indexes)   # selected indeces contains the list of active AEs
    RMSEsSelectedAE.append(RMSE)


# In this example, we only have beneign samples, so one sample is intentionally classified as 1 in order to get ROC/AUC 
L_B_train[100] = 1      

# Computing AUC Scores
aucScores=roc_auc_score(L_B_train,RMSEsSelectedAE)
# plotting roc curve
fpr, tpr, thresholds = roc_curve(L_B_train,RMSEsSelectedAE)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.5f' % aucScores)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
# plt.savefig('title.png')     # to save the image


# Calculating Performance metrices
threshold = bestThreshold(tpr,fpr,thresholds)
predicted = predict(RMSEsSelectedAE,threshold)
bestT     = bestThreshold(tpr,fpr,thresholds)
np.set_printoptions(precision=2)
# Plot non-normalized confusion matrix
plot_confusion_matrix(L_B_train, predicted,classes=["benign","malicious"],
                      title='Confusion matrix(Video Injection):Selected AEs (0.5 of original ensemble)')
plt.show()
# plt.savefig('title.png')


# Plot of Mean RMSEs
sorted_values = [rmse for _, rmse in sorted_rmse]
original_indexes = [index for index, _ in sorted_rmse]
plt.figure(figsize=(10, 6))
plt.plot(original_indexes, sorted_values , marker='o', linestyle='-', color='b', label='RMSE')
plt.xlabel('Index of Active Autoencoder')
plt.ylabel('RMSE')
plt.title('Line Plot of Mean RMSEs')
plt.grid(True)
plt.tight_layout()
plt.legend()
plt.show()
# plt.savefig('title.png')

    
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