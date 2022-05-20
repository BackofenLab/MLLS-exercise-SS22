# -*- coding: utf-8 -*-
import numpy as np
import networkx as nx
from eden.graph import Vectorizer
from sklearn.metrics import pairwise
from scipy.sparse import vstack
from sklearn import svm
import glob
from sklearn.model_selection import StratifiedKFold
from sklearn import metrics
from sklearn.model_selection import GridSearchCV
import pandas as pd

def create_graphs(arry):
    """
    Parameters:
    - file_path: path to adjacency matrices
    - start_ID: Starting ID for graph nodes
    
    Return: An undirected, unweighted graph
    """

    start_ID = 0
    graphs = []
    for AM in [arry]:
        N = AM.shape[0]  
        g = nx.Graph()    
       
        g.add_nodes_from(range(start_ID, N+start_ID), label="")
        list_edges = []
        for u in range(N-1):
            for v in range(u+1,N):
                w = AM[u,v]
                if w != 0.0:
                    list_edges.append((u+start_ID, v+start_ID))
        g.add_edges_from(list_edges, label = "")
        graphs.append(g)
        
        start_ID+= start_ID + N
            
    return graphs



def NSPDK(g, d, r):
    """ Vectorize graph nodes
    
    Needs graph input

    Return: a matrix in which rows are the vectors that represents for nodes         
    """

    vec = Vectorizer(nbits=16, 
		         discrete=True, 
		         d=d,
		         r=r,
		         )
		     
    print(vec)
		         
    M = vec.vertex_transform(g)
    print(M)
    M = M[0] 
    K = pairwise.linear_kernel(M,M)
         
    return K   
	
	
	
	
def get_RLK(A, alpha=4.):
    """ Compute regularized Laplacian kernel.
    
    Parameters:
        A -- Adjacency matrix.
        alpha -- Diffusion parameter (positive float, 4.0 by default).
        
    Return:
        RLK -- Regularized Laplacian kernel matrix.
    """
    
    from scipy.linalg import inv
    
    # N is the number of vertices
    N = A.shape[0]
    for idx in range(N):
        A[idx, idx] = 0
    
    I = np.identity(N)
    D = np.zeros((N,N))
    for idx in range(N):
        D[idx,idx] = sum(A[idx,:])
    L = D - A
    RLK = inv(I + alpha*L)
    
    return RLK

def box_plot(c_eval, x_tick_labels):

    fig1, ax1 = plt.subplots()   
    ax1.boxplot(c_eval)
    ax1.set_xticklabels(x_tick_labels)
    ax1.set_title('C-param')
    ax1.set_ylabel('performance')
    plt.show()

    return

def get_RBF(A, s=1.):
    """ Compute radial basis function kernel.
    
    Parameters:
        A -- Feature matrix.
        s -- Scale parameter (positive float, 1.0 by default).
        
    Return:
        K -- Radial basis function kernel matrix.
    """
    
    from sklearn.metrics.pairwise import euclidean_distances, rbf_kernel
    from sklearn.preprocessing import scale
    
    A = scale(A)
    #dist_matrix = euclidean_distances(A, A, None, squared=True)
    dist_matrix = euclidean_distances(A, None, squared=True)
    dist_vector = dist_matrix[np.nonzero(np.tril(dist_matrix))]
    dist_median = np.median(dist_vector)
    K = rbf_kernel(A, None, dist_median*s)
    
    return K
    
    
    

def get_MEDK(A, beta=0.04):
    """ Compute Markov exponential diffusion kernel.
    
    Parameters:
        A -- Adjacency matrix.
        beta -- Diffusion parameter (positive float, 0.04 by default).
        
    Return:
        MEDK -- Markov exponential diffusion kernel matrix.
    """
    
    from cvxopt import matrix
    from scipy.linalg import expm
    
    # N is the number of vertices        
    N = A.shape[0]
    for idx in range(N):
        A[idx,idx] = 0
    A = matrix(A)
    D = np.zeros((N,N))
    for idx in range(N):
        D[idx, idx] = sum(A[idx,:])
    I = np.identity(N)
    M = (beta/N)*(N*I - D + A)
    MEDK = expm(M)
    
    return MEDK




    

	
def main():


    
    
    return


	
if __name__=="__main__":


    main()


    
