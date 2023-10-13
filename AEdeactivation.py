from myKitsune import Kitsune
import time
import numpy as np
import pandas as pd

Df=pd.read_csv("Mirai_dataset.csv",names=list(range(1,116)))
packets,features=Df.shape

# KitNET params:
maxAE = 10 #maximum size for any autoencoder in the ensemble layer
FMgrace = 5000 #the number of instances taken to learn the feature mapping (the ensemble's architecture)
ADgrace = 50000 #the number of instances used to train the anomaly detector (ensemble itself)

# Build Kitsune

K = Kitsune(features,maxAE,FMgrace,ADgrace)

print("Running Kitsune:")
RMSEs = []
start = time.time()

for i in range(packets):
    
    if i==100000: # number of packets to process
        break

    r=Df.loc[i].to_numpy()
    predictions = K.proc_next_packet(r)
    RMSEs.append(predictions)

    if i % 1000 == 0:
        print(i)
        print(RMSEs[i])

stop = time.time()
print("Complete. Time elapsed: "+ str(stop - start))