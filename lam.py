import numpy as np
samples_per_cls=[9914, 5770, 737, 586]
beta =0.999
effective_num = 1.0-np.power(beta, samples_per_cls)
weights = (1.0 - beta) / np.array(effective_num)
print (weights)