from KitNET.KitNET import KitNET


class Kitsune:
    def __init__(self,n,max_autoencoder_size=10,FM_grace_period=None,AD_grace_period=10000,learning_rate=0.1,hidden_ratio=0.75):
       self.AnomDetector = KitNET(n,max_autoencoder_size,FM_grace_period,AD_grace_period,learning_rate,hidden_ratio)

    def proc_next_packet(self,x,activeDA = None):
    
        # process KitNET
        return self.AnomDetector.process(x,activeDA )  # will train during the grace periods, then execute on all the rest.
         