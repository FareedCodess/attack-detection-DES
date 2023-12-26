#%%
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.model_selection import train_test_split
from myKitsune import Kitsune
import time
import numpy as np
import pandas as pd
from train_test_module import getDistinctDatasets,train,test,bestThreshold,plot_confusion_matrix,predict
import matplotlib.pyplot as plt


# Legend for variables:
# L=labels ; B= benign ; M= malicious 

# loading packet features and its labels
packetsDf=pd.read_csv("Kitsune Dataset from Kaggle/Video Injection/Video_Injection_dataset.csv",names=list(range(1,116)))
labelDf=pd.read_csv("Kitsune Dataset from Kaggle/Video Injection/Video_Injection_labels.csv",names=['x','labels'])
labelDf=labelDf.drop(0)
labelDf.drop(columns='x',inplace=True)
labelDf=labelDf.astype(np.int64)
labelDf=labelDf.reset_index(drop=True)

# filtering benign and malicious samples
benignDf,maliciousDf=getDistinctDatasets(packetsDf,labelDf['labels'])

# splitting benign dataset int two folds of test size 0.4
B_train, B_test, L_B_train, L_B_test = train_test_split(benignDf[list(range(1,116))].to_numpy(),benignDf['labels'].to_numpy(), test_size=0.8 ,shuffle= False,random_state=42)

# training kitsune on beniegn  dataset
# train(B_train)
packets,features=B_train.shape

# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 5000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 50000 #the number of instances used to train the anomaly detector (ensemble itself)

# Build Kitsune
K = Kitsune(features,maxAE,FMgrace,ADgrace)

print("Running Kitsune:")

RMSEs = []
fullEnsembleRMSEs = []
print("Training phase :")
for z in range(55000):
    r = B_train[z]
    if z % 10000 == 0:
        print(z)
    K.proc_next_packet(r)


print("Collecting RMSEs for Full Ensemble")
for z in range(55000):
    r = B_train[z]
    if z % 10000 == 0:
        print(z)
    fullEnsembleRMSEs.append(K.proc_next_packet(r))


print("Testing on training data by activating one autoencoder.") 
for i in range(len(K.AnomDetector.ensembleLayer)):
    RMSEsOneAE =[]
    print (f"Current Active AE : {i}")
    for j in range(55000):
        r=B_train[j]
        RMSE = K.proc_next_packet(r,i)
        RMSEsOneAE.append(RMSE)

        if j % 10000 == 0:
            print(j)
            # print(RMSEs[i])
    RMSEs.append(RMSEsOneAE)

FE_arr = np.array(fullEnsembleRMSEs)
FE_filtered_arr = FE_arr[FE_arr <= 1]
FE_mean_of_RMSE = np.mean(FE_filtered_arr)
MeanRMSEs = []
for i in RMSEs:
    arr = np.array(i)
    # Remove elements greater than 10
    filtered_arr = arr[arr <= 1]
    # Calculate the mean of the remaining elements
    mean_of_filtered = np.mean(filtered_arr)
    MeanRMSEs.append(mean_of_filtered)

indexed_rmse = list(enumerate(MeanRMSEs))
sorted_rmse = sorted(indexed_rmse, key=lambda x: x[1],reverse=False)
# sorted_rmse.insert(0,('FE',FE_mean_of_RMSE))

sorted_values = [rmse for _, rmse in sorted_rmse]
original_indexes = [index for index, _ in sorted_rmse]


selected_indexes = [original_indexes[i] for i in range(len(original_indexes)//2)]
print("Selected Autoencoders: ", selected_indexes)
RMSEsSelectedAE =[]
print("Testing on training data by activating selected autoencoder.") 
for i in range(55000):
    if i % 10000 == 0:
        print(i)
    r = B_train[i]
    RMSE = K.proc_next_packet(r,selected_indexes)
    RMSEsSelectedAE.append(RMSE)

#%% 
B_train, B_test, L_B_train, L_B_test = train_test_split(benignDf[list(range(1,116))].to_numpy(),benignDf['labels'].to_numpy(), test_size=0.8 ,shuffle= False,random_state=42)

L_B_train[100] = 1
L_B_train = L_B_train[0:55000]
aucScores=roc_auc_score(L_B_train,RMSEsSelectedAE)
#plotting roc curve
fpr, tpr, thresholds = roc_curve(L_B_train,RMSEsSelectedAE)
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', label = 'AUC = %0.5f' % aucScores)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.savefig('Video_Injection_A20.png')

## calculating score
threshold=bestThreshold(tpr,fpr,thresholds)
predicted=predict(RMSEsSelectedAE,threshold)

# bestT=bestThreshold(tpr,fpr,thresholds)
# print(log_loss(L_BM_Test1[0:50000],RMSEs))
# print(score(RMSEs,L_BM_Test1[0:50000],bestT))

np.set_printoptions(precision=2)

# Plot non-normalized confusion matrix
# plot_confusion_matrix(L_BM_Test1, predicted,classes=["benign","malicious"],title='Confusion matrix(Video Injection): Deactivating 50% Autoendcoders')
plot_confusion_matrix(L_B_train, predicted,classes=["benign","malicious"],title='Confusion matrix(Video Injection):Selected AEs (0.5 of original ensemble)')

# # Plot normalized confusion matrix
# plot_confusion_matrix(, labels, normalize=True,
#               title='Normalized confusion matrix')
plt.savefig('confusionMatrix_Video_Injection_selectedAEs_order_bestPerforming(RMSE).png')



#%%
plt.figure(figsize=(10, 6))
plt.plot(original_indexes, sorted_values , marker='o', linestyle='-', color='b', label='RMSE')
plt.xlabel('Index of Active Autoencoder')
plt.ylabel('RMSE')
plt.title('Line Plot of Mean RMSEs')
plt.grid(True)
plt.tight_layout()
plt.legend()

plt.savefig('line_plot_55K_VI_ZeroFeed.png')
print("Complete.") 


# %%

for i in RMSEs:
    arr = np.array(i)

    # Remove elements greater than 10
    filtered_arr = arr[arr <= 1]
    print(filtered_arr.size)
    # Calculate the mean of the remaining elements
    mean_of_filtered = np.mean(filtered_arr)
    MeanRMSEs.append(mean_of_filtered)

# %%
    
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
