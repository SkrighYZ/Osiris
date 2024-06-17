import numpy as np

def accuracy(accs):
    # assumes a uniform distribution p(t) over tasks, hence the mean
    return accs[-1].mean() 

def forgetting(accs):
    T = accs.shape[0]
    total = 0
    for i in range(T-1):
        total += (accs[:, i].max() - accs[-1, i])
    return total / (T - 1)

def knowledge_gain(accs):
    T = accs.shape[0]
    total = 0
    for i in range(T-1):
        total += accs[i+1, i+1] - accs[i, i+1]
    return total / (T - 1)

def consolidation(taccs):
    return taccs[-1]

def forward_transfer(accs, random_accs):
    T = accs.shape[0]
    total = 0
    for i in range(T-1):
        total += accs[i, i+1] - random_accs[i+1]
    return total / (T - 1)