import numpy as np
import pandas as pd
from taucc.taucc import CoClust
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import issparse
from sklearn.utils import check_array
from time import time

class HCoClust():

    def __init__(self, max_row_clusters=None, max_col_clusters=None, max_row_level=None, max_col_level=None, n_iterations=500, n_iter_per_mode = 100, initialization= 'random', k = 30, l = 30, row_clusters = np.zeros(1), col_clusters = np.zeros(1), initial_prototypes = np.zeros(1), verbose = False, random_state=None) -> None:
        self.max_row_clusters = max_row_clusters
        self.max_col_clusters = max_col_clusters
        self.max_row_level = max_row_level
        self.max_col_level = max_col_level
        self.n_iterations = n_iterations
        self.n_iter_per_mode = n_iter_per_mode
        self.initialization = initialization
        self.k = k
        self.l = l
        self.row_clusters = row_clusters
        self.col_clusters = col_clusters
        self.initial_prototypes = initial_prototypes
        self.verbose = verbose
        self.labelencoder_ = LabelEncoder()
        self.rng = np.random.default_rng(seed = random_state)
        # these fields will be available after calling fit
        self.row_levels_ = 0
        self.col_levels_ = 0
        self.row_labels_ = None
        self.column_labels_ = None
        self.execution_time_ = None
        np.seterr(all='ignore')

    def _init_all(self, V):
        self._dataset = None
        self._dataset = check_array(V, accept_sparse='csr', dtype=[np.float64, np.float32, np.int32])
        self._csc_dataset = None
        if issparse(self._dataset):
            # transform also to csc
            self._csc_dataset = self._dataset.tocsc()
            
        # the number of documents and the number of features in the data (n_rows and n_columns)
        self._n_documents = self._dataset.shape[0]
        self._n_features = self._dataset.shape[1]

        # the number of row/ column clusters
        self._n_row_clusters = []
        self._n_col_clusters = []

        # a list of n_documents (n_features) elements
        # for each document (feature) d contains the row cluster index d is associated to
        self._row_assignment = []
        self._col_assignment = []
        self._tmp_row_assignment = np.zeros(self._n_documents)
        self._tmp_col_assignment = np.zeros(self._n_features)
        self._tot = np.sum(self._dataset)
        self._dataset = self._dataset/self._tot
        self.tau_x = []
        self.tau_y = []
        model = CoClust(self.n_iterations, self.n_iter_per_mode, self.initialization, min(self.k, self._n_documents), min(self.l, self._n_features), self.row_clusters, self.col_clusters, self.initial_prototypes, self.verbose, self.rng)
        self._start_time = time()
        model.fit(V)
        self._row_assignment.append(model.row_labels_)
        self._col_assignment.append(model.column_labels_)
        self._n_row_clusters.append(len(np.unique(self._row_assignment)))
        self._n_col_clusters.append(len(np.unique(self._col_assignment)))
        self._row_incidence = model._row_incidence
        self._col_incidence = model._col_incidence
        self.tau_x.append(model.tau_x[-1])
        self.tau_y.append(model.tau_y[-1])
        self.row_levels_ = 1
        self.col_levels_ = 1
        self._T = model._T


    def fit(self, V):
        self._init_all(V)
        finished_row = False
        finished_col = False
        while (not(finished_row) or not(finished_col)):         
            if (self.max_row_level is not None):
                if (self.max_row_level <= self.row_levels_):
                    finished_row = True
            elif (self.max_row_clusters is not None):
                if (self.max_row_clusters <= self._n_row_clusters[-1]):
                    finished_row = True
            elif (self._n_row_clusters[-1]==self._n_documents):
                finished_row = True
            if (self.max_col_level is not None):
                if (self.max_col_level <= self.col_levels_):
                    finished_col = True
            elif (self.max_col_clusters is not None):
                if (self.max_col_clusters <= self._n_col_clusters[-1]):
                    finished_col = True
            elif (self._n_col_clusters[-1]==self._n_features):
                finished_col = True
            if (finished_row is not True):
                start_clust = 0
                for i in range(self._n_row_clusters[-1]):
                    clust_index = np.where(np.array(self._row_assignment[-1])==i)[0]
                    if (len(clust_index)==1):
                        self._tmp_row_assignment[clust_index]=start_clust
                    else:
                        R = self._dataset[clust_index]
                        self._random_initialization(0,clust_index)
                        actual_iteration_x = 0    
                        contr = True
                        while contr:
                            contr = self._perform_row_move(R,clust_index)
                            actual_iteration_x += 1
                            if actual_iteration_x > self.n_iter_per_mode:
                                contr = False
                        self._adjust_cluster_label(0,clust_index,start_clust)
                    start_clust += len(np.unique(self._tmp_row_assignment[clust_index]))
                self._check_clustering_final(0)
                self.row_levels_ +=1
            if (finished_col is not True):
                start_clust = 0
                for i in range(self._n_col_clusters[-1]):
                    clust_index = np.where(np.array(self._col_assignment[-1])==i)[0]
                    if (len(clust_index)==1):
                        self._tmp_col_assignment[clust_index]=start_clust
                    else:
                        C = self._dataset[:,clust_index]
                        self._random_initialization(1,clust_index)
                        actual_iteration_y = 0    
                        contc = True
                        while contc:
                            contc = self._perform_col_move(C, clust_index)
                            actual_iteration_y += 1
                            if actual_iteration_y > self.n_iter_per_mode:
                                contc = False
                        self._adjust_cluster_label(1,clust_index,start_clust)
                    start_clust += len(np.unique(self._tmp_col_assignment[clust_index]))
                self._check_clustering_final(1)
                self.col_levels_ +=1
            if (not finished_col) and (finished_row or len(np.unique(self._row_assignment[-1]))==len(np.unique(self._row_assignment[-2]))):
                if (len(np.unique(self._col_assignment[-1]))==len(np.unique(self._col_assignment[-2]))):
                    finished_col=True
            if (not finished_row) and (finished_col or (len(np.unique(self._col_assignment[-1]))==len(np.unique(self._col_assignment[-2])))):
                if len(np.unique(self._row_assignment[-1]))==len(np.unique(self._row_assignment[-2])):
                    finished_row = True
            self._tmp_row_incidence=self._row_incidence
            self._tmp_col_incidence=self._col_incidence
            self._T = self._init_contingency_matrix(self._dataset,1)[1]
            tau_x, tau_y = self.compute_taus()
            self.tau_x.append(tau_x)
            self.tau_y.append(tau_y)

        # clone cluster assignments and transform in lists
        self.row_labels_ = np.copy(self._row_assignment[-1]).tolist()
        self.column_labels_ = np.copy(self._col_assignment[-1]).tolist()                   
        self._end_time = time()
        self.execution_time_ = self._end_time - self._start_time

    def _random_initialization(self, dimension, clust_index):
        if (dimension == 0):
            self._tmp_row_assignment[clust_index] = self.rng.choice(min(self.k,len(clust_index)), size = len(clust_index))
            self._check_clustering(dimension,clust_index)
            self._tmp_row_incidence = np.zeros((len(clust_index), len(np.unique(self._tmp_row_assignment[clust_index]))))
            self._tmp_row_incidence[np.arange(0,len(clust_index),dtype='int'), self._tmp_row_assignment[clust_index].astype(int)] = 1 
        if (dimension == 1):
            self._tmp_col_assignment[clust_index] = self.rng.choice(min(self.l,len(clust_index)), size = len(clust_index))
            self._check_clustering(dimension,clust_index)
            self._tmp_col_incidence = np.zeros((len(clust_index), len(np.unique(self._tmp_col_assignment[clust_index]))))
            self._tmp_col_incidence[np.arange(0,len(clust_index),dtype='int'), self._tmp_col_assignment[clust_index].astype(int)] = 1 
        #self._check_clustering(dimension,clust_index,start_clust)
        

    def _adjust_cluster_label(self, dimension, clust_index, start_cluster):
        if dimension == 1:
            self._tmp_col_assignment[clust_index] += start_cluster
        elif dimension == 0:
            self._tmp_row_assignment[clust_index] += start_cluster 


    def _check_clustering(self, dimension, clust_index):
        if dimension == 1:
            self._tmp_col_assignment[clust_index] = self.labelencoder_.fit_transform(self._tmp_col_assignment[clust_index].astype(int))
            self._tmp_col_incidence = np.zeros((len(clust_index), len(np.unique(self._tmp_col_assignment[clust_index]))))
            self._tmp_col_incidence[np.arange(0,len(clust_index),dtype='int'), self._tmp_col_assignment[clust_index].astype(int)] = 1 
        elif dimension == 0:
            self._tmp_row_assignment[clust_index] = self.labelencoder_.fit_transform(self._tmp_row_assignment[clust_index].astype(int))
            self._tmp_row_incidence = np.zeros((len(clust_index), len(np.unique(self._tmp_row_assignment[clust_index]))))
            self._tmp_row_incidence[np.arange(0,len(clust_index),dtype='int'), self._tmp_row_assignment[clust_index].astype(int)] = 1 

    def _check_clustering_final(self, dimension):
        if dimension == 1:
            self._col_assignment.append(np.copy(self._tmp_col_assignment))
            self._n_col_clusters.append(len(np.unique(self._tmp_col_assignment)))
            self._col_incidence = np.zeros((self._n_features, self._n_col_clusters[-1]))      
            self._col_incidence[np.arange(0,self._n_features,dtype='int'), self._tmp_col_assignment.astype(int)] = 1     
        elif dimension == 0:
            self._row_assignment.append(np.copy(self._tmp_row_assignment))
            self._n_row_clusters.append(len(np.unique(self._tmp_row_assignment)))
            self._row_incidence = np.zeros((self._n_documents, self._n_row_clusters[-1]))
            self._row_incidence[np.arange(0,self._n_documents,dtype='int'), self._tmp_row_assignment.astype(int)] = 1 

    def _init_contingency_matrix(self, V, dimension):
        dataset = self._update_dataset(V,dimension)
        #new_t = np.zeros((self._n_row_clusters, self._n_col_clusters), dtype=float)
        if dimension == 0:
            new_t = np.dot(self._tmp_row_incidence.T, dataset)
        else:
            new_t = np.dot(dataset, self._tmp_col_incidence)   
        return dataset, new_t

    def _update_dataset(self, V, dimension):
        if dimension == 0:
            #new_t = np.zeros((self._n_documents, self._n_col_clusters), dtype = float)
            new_t = np.dot(V, self._col_incidence)             
        else:
            #new_t = np.zeros((self._n_row_clusters, self._n_features), dtype = float)
            new_t = np.dot(self._row_incidence.T, V)
        return new_t


    def _perform_row_move(self, V, clust_index):
        dataset, T = self._init_contingency_matrix(V,0)
        tmp_row_assignment = np.zeros(np.shape(V)[0])
        S = np.repeat(np.sum(T, axis = 1).reshape((-1,1)), repeats = T.shape[1], axis = 1)
        B = T/np.sum(T, axis = 0) - S
        moves = 0
        all_tau = np.dot(dataset,B.T)
        max_tau = np.max(all_tau, axis = 1)
        e_max = np.where(max_tau == all_tau.T)
        tmp_row_assignment[e_max[1][:np.shape(V)[0]]] = e_max[0][:np.shape(V)[0]]
        moves = np.sum(tmp_row_assignment != self._tmp_row_assignment[clust_index])
        if moves > 0:
            self._tmp_row_assignment[clust_index] = tmp_row_assignment
            self._check_clustering(0, clust_index)
        if moves:
            return True
        else:
            return False
        
    def _perform_col_move(self, V, clust_index):

        dataset, T = self._init_contingency_matrix(V,1)
        tmp_col_assignment = np.zeros(np.shape(V)[1])
        T = T.T
        dataset = dataset.T
        S = np.repeat(np.sum(T, axis = 1).reshape((-1,1)), repeats = T.shape[1], axis = 1)
        B = T/np.sum(T, axis = 0) - S
        moves = 0
        all_tau = np.dot(dataset,B.T)
        max_tau = np.max(all_tau, axis = 1)
        e_max = np.where(max_tau == all_tau.T)
        tmp_col_assignment[e_max[1][:np.shape(V)[1]]] = e_max[0][:np.shape(V)[1]]
        moves = np.sum(tmp_col_assignment != self._tmp_col_assignment[clust_index])
        if moves > 0:
            self._tmp_col_assignment[clust_index] = tmp_col_assignment
            self._check_clustering(1, clust_index)
        if moves:
            return True
        else:
            return False 
        

    def compute_taus(self):
        tot_per_x = np.sum(self._T, 1)
        tot_per_y = np.sum(self._T, 0)
        t_square = np.power(self._T, 2)

        a_x = np.sum(np.nan_to_num(np.true_divide(np.sum(t_square, axis = 0), tot_per_y)))
        b_x = np.sum(np.power(tot_per_x, 2))
        

        a_y = np.sum(np.nan_to_num(np.true_divide(np.sum(t_square, axis = 1), tot_per_x)))
        b_y = np.sum(np.power(tot_per_y, 2))

        tau_x = np.nan_to_num(np.true_divide(a_x - b_x, 1 - b_x))
        tau_y = np.nan_to_num(np.true_divide(a_y - b_y, 1 - b_y))

        return tau_x, tau_y#, a_x, b_x