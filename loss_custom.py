import torch
import torch.nn as nn
import numpy as np

m = nn.Softmax(dim=1)
input = torch.randn(2, 3)
n=m(input)
def cross_entropy(inputs,labels):
    m = nn.Softmax(dim=1)
    n = m(inputs)
    loss=-torch.sum(labels*torch.log(n))
    return loss
def class_loss(inputs,labels,beta,samples_per_cls,no_of_classes):
    effective_num = 1.0 - np.power(beta, samples_per_cls)
    weights = (1.0 - beta) / np.array(effective_num)
    weights = weights / np.sum(weights) * no_of_classes
    weights = torch.tensor(weights).float().to("cuda")
    cb_loss=weights*cross_entropy(inputs,labels)
    return cb_loss

