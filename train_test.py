#%%
import pandas as pd
import time
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score,roc_curve,accuracy_score,log_loss
from train_test_module import getDistinctDatasets,train,test,bestThreshold,plot_confusion_matrix,predict
from KitNET.dA import dA




# Legend for variables:
# L=labels ; B= benign ; M= malicious 

# loading packet features and its labels
packetsDf=pd.read_csv("Kitsune Dataset from Kaggle/Video Injection/Video_Injection_dataset.csv",names=list(range(1,116)))
labelDf=pd.read_csv("Kitsune Dataset from Kaggle/Video Injection/Video_Injection_labels.csv",names=['x','labels'])
labelDf=labelDf.drop(0)
labelDf=labelDf.drop(columns='x')
labelDf=labelDf.astype(np.int64)
labelDf=labelDf.reset_index(drop=True)

# filtering benign and malicious samples
benignDf,maliciousDf=getDistinctDatasets(packetsDf,labelDf['labels'])

# splitting benign dataset int two folds of test size 0.4
B_train, B_test, L_B_train, L_B_test = train_test_split(benignDf[list(range(1,116))].to_numpy(),benignDf['labels'].to_numpy(), test_size=0.8 ,shuffle= False,random_state=42)

modelName = 'VI.pkl'
#training kitsune on beniegn  dataset
B_RMSEs = train(B_train,modelName)


#creating a testset which has both malicious and unseen benign samples
# B_testDf =pd.DataFrame(B_test,columns=list(range(1,116)))
# L_B_testDf =pd.DataFrame(L_B_test,columns=['labels'])
# benignTestDf =pd.concat([B_testDf,L_B_testDf],axis=1)
# BM_testDf=pd.concat([maliciousDf,benignTestDf])

# BM_Test1 , BM_Test2 , L_BM_Test1 , L_BM_Test2 = train_test_split(BM_testDf[list(range(1,116))].to_numpy() , BM_testDf['labels'].to_numpy(), train_size=0.9999999 ,shuffle= True)
# startTime=time.time()
# B_M_RMSEs=test(BM_Test1,modelName)
# endTime=time.time()
#%%
x = np.array(B_RMSEs)
# Count how many elements are above 10
count_above_10 = np.count_nonzero(x > 1)

print(f"The number of elements above 10 is: {count_above_10}")
indexes_above_10 = np.where(x > 1)

print("Indexes of elements above 10:", indexes_above_10)


#%%
# print("Time taken to test %d packets is %lf."%(len(BM_Test1),float(endTime-startTime)))

# aucScores=roc_auc_score(L_BM_Test1,RMSEs)

# import csv 
# obj=open("auc_20.csv",'a',newline='')
# wr=csv.writer(obj)
# wr.writerow(["Video Injection_A20",aucScores,len(BM_Test1),endTime-startTime])



# #plotting roc curve
# fpr, tpr, thresholds = roc_curve(L_BM_Test1,RMSEs)
# plt.title('Receiver Operating Characteristic')
# plt.plot(fpr, tpr, 'b', label = 'AUC = %0.5f' % aucScores)
# plt.legend(loc = 'lower right')
# plt.plot([0, 1], [0, 1],'r--')
# plt.xlim([0, 1])
# plt.ylim([0, 1])
# plt.ylabel('True Positive Rate')
# plt.xlabel('False Positive Rate')
# plt.savefig('Video_Injection_A20.png')

# ## calculating score
# thres=bestThreshold(tpr,fpr,thresholds)
# predicted=predict(RMSEs,thres)

# # bestT=bestThreshold(tpr,fpr,thresholds)
# # print(log_loss(L_BM_Test1[0:50000],RMSEs))
# # print(score(RMSEs,L_BM_Test1[0:50000],bestT))

# np.set_printoptions(precision=2)

# # Plot non-normalized confusion matrix
# # plot_confusion_matrix(L_BM_Test1, predicted,classes=["benign","malicious"],title='Confusion matrix(Fuzzing): Deactivating 50% Autoendcoders')
# plot_confusion_matrix(L_BM_Test1, predicted,classes=["benign","malicious"],title='Confusion matrix(Video Injection):All AEs Activated')

# # # Plot normalized confusion matrix
# # plot_confusion_matrix(, labels, normalize=True,
# #                       title='Normalized confusion matrix')

# plt.savefig('confusionMatrix_Video_Injection.png')