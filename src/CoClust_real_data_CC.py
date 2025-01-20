import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score as ari
from sklearn.metrics import normalized_mutual_info_score as nmi
import matplotlib.pyplot as plt

from algorithms.coclust_DP import CoClust

from utils import CreateOutputFile, execute_test_dp, execute_test_cc
import sys


#dataset = 'cstr'
datasets = ['classic3', 'cstr', 'hitech', 'k1b', 'reviews', 'sports', 'tr11', 'tr23', 'tr41', 'tr45']
n_test = 10
#k = 4


for dataset in datasets:
        
    dt = pd.read_csv(f'./data/{dataset}.txt')
    t = pd.read_csv(f'./data/{dataset}_target.txt', header = None)
    target = np.array(t).T[0]


    n = len(dt.doc.unique())
    m = len(dt.word.unique())
    k = len(t[0].unique())
    print(k)
    T = np.zeros((n,m), dtype = int)


    for g in dt.iterrows():
        T[g[1].doc,g[1].word] = g[1].cluster
        #T[g[1].doc,g[1].word] = 1

    f, date = CreateOutputFile(dataset)
    ty = np.zeros(m, dtype = int)
    for t in range(n_test):
        model = execute_test_cc(f, T, [target, ty], noise=0,n_iterations = 3, init = [k+10,k+10], verbose = False)
        
    f.close()
