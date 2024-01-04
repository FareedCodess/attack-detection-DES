# Artificial Intelligence for Intrusion Detection in IoT
Neural networks have grown in popularity as a solution for network intrusion detection systems (NIDS). Because of their ability to learn complicated patterns and behaviors, they are an ideal solution for distinguishing between normal traffic and network attacks. 

However, one disadvantage of neural networks is the large number of resources required to train them. Many network gateways and routers, which can potentially host a NIDS, lack the memory or computing power to train and execute such models. One approach to tackle this problem has been proposed, which is to use a plug-and-play NIDS that can learn to detect attacks on the local network, without supervision in an efficient manner. 

This project lays the groundwork for potentially improving anomaly detection accuracy by exploring the feasibility of embedding a dynamic ensemble selection classifier layer within the neural networks. While a fully dynamic layer is still under development, we have successfully implemented a static version that demonstrates the potential of this approach. Our findings provide valuable insights for future research and development efforts focused on realizing the dynamic capability.

![Error loading image](/Time_50.png)
The above figure shows the time comparison of deactivating 50% of autoencoders in the ensemble layer highlighting the potential to reduce energy consumption. 
To learn more, please refer to 'ProjectReport.pdf'.

Dataset used for the project :
Kitsune Network Attack Dataset. (2019). UCI Machine Learning Repository. https://doi.org/10.24432/C5D90Q.


