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

    Return: a matrix in which rows are the vectors that represents for nodes         
    """

    vec = Vectorizer(nbits=16, 
		         discrete=True, 
		         d=d,
		         r=r,
		         )
	
	
    M = vec.vertex_transform(g)
    
    
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
    
    
    
    
def get_kernel_matrix(kernel_type, adjacency_matrix):

    if kernel_type == "RLK":
       kernel_matrix = get_RLK(adjacency_matrix)
    elif kernel_type == "MEDK":
        kernel_matrix = get_MEDK(adjacency_matrix)
    elif kernel_type == "RBF":
        kernel_matrix = get_RBF(adjacency_matrix)
    elif kernel_type == "NSPDK":
        graphs = create_graphs(adjacency_matrix)
        kernel_matrix = NSPDK(graphs, d=2, r=2)
    else:
        print("kernel not implemented")
        raise NotImplementedError

    return kernel_matrix
    
    
    
def GridSearch(parameter1, parameter2, adjacency_matrix, combined_indeces, y_value):

	grid_search_eval = []
	grid_search_best_parameter = []
  
	outer_k_fold = StratifiedKFold(n_splits = 3, random_state=None, shuffle=True)         
	for enum, indeces in enumerate(outer_k_fold.split(range(len(y_value)), y_value)):
		inner_index, outer_index = indeces[0], indeces[1]
    

		hyp_res = []
		search_parameter = [(x,y) for x in parameter1 for y in parameter2] 
		last_kernel = None
    	

		for parameter_num, parameter_pair in enumerate(search_parameter):
         

			if parameter_pair[0] != last_kernel: 
				kernel_matrix = get_kernel_matrix(parameter_pair[0], adjacency_matrix)
				selected_kernel_matrix = kernel_matrix[np.ix_(combined_indeces, combined_indeces)]

			k_fold = StratifiedKFold(n_splits = 5, random_state=None, shuffle=True) 
			clf_res = []
	            
			inner_train, outer_test = selected_kernel_matrix[np.ix_(inner_index, inner_index)], selected_kernel_matrix[np.ix_(outer_index, inner_index)]
			inner_y_train, outer_y_test = y_value[inner_index], y_value[outer_index]
        
			for num, indeces in enumerate(k_fold.split(range(len(inner_y_train)), inner_y_train)):
			
    
				train_index, test_index = indeces[0], indeces[1]
				X_train, X_test = inner_train[np.ix_(train_index, train_index)], inner_train[np.ix_(test_index, train_index)]
				y_train, y_test = inner_y_train[train_index], inner_y_train[test_index]
				clf = svm.SVC(C = parameter_pair[1], kernel='precomputed')
				clf.fit(X_train,y_train)
				res = clf.predict(X_test)
				clf_res.append(metrics.roc_auc_score(y_test, res))
				
				
				
			hyp_res.append(np.mean(clf_res))
			last_kernel = parameter_pair[0]
	        

		###### final evaluation #####
		best_parameter = search_parameter[np.argmax(hyp_res)]
		print("###################################################")
		print(f"Best parameter in fold {enum}: {best_parameter}")        
		final_eval = evaluate_graph_kernel(best_parameter[0], best_parameter[1],adjacency_matrix, y_value, inner_index, outer_index, combined_indeces)
		grid_search_eval.append(final_eval)
		grid_search_best_parameter.append(best_parameter)
		#############################
	

	print(f"Resuts of nested CV grid search: {np.mean(grid_search_eval)}:")        

	return grid_search_eval, grid_search_best_parameter



def evaluate_graph_kernel(kernel_type, C_param,  adjacency_matrix, y_value, inner_index, outer_index, combined_indeces):
    
    """ Evaluates graph kernel on dataset
    
    Parameters:
        kernel_type -- graph kernel to be used.
        adjacency_matrix -- Adjacency matrix.
        combined_indeces -- indeces of dataset
        y_value -- class value
        
    Return:
        clf_res -- classification result.
    """
	
    kernel_matrix = get_kernel_matrix(kernel_type, adjacency_matrix)
    selected_kernel_matrix = kernel_matrix[np.ix_(combined_indeces, combined_indeces)]
    inner_train, outer_test = selected_kernel_matrix[np.ix_(inner_index, inner_index)], selected_kernel_matrix[np.ix_(outer_index, inner_index)]
    inner_y_train, outer_y_test = y_value[inner_index], y_value[outer_index]
        
	
	

    clf = svm.SVC(C = C_param, kernel='precomputed')
    clf.fit(inner_train,inner_y_train)
    res = clf.predict(outer_test)


    print("#############################################################")
    print("ROC-AUC " + kernel_type + f"-kernel: {metrics.roc_auc_score(outer_y_test, res)}")
    print("#############################################################")

    return metrics.roc_auc_score(outer_y_test, res)






	
def main():


    adjacency_matrix = np.loadtxt("./adjacency_matrices/hprd_int")
    index_file = "./all_genes"
    index_file_open = open(index_file)
    index_file_lines = index_file_open.readlines()
    index_file_open.close()
    gene_list = [gene[:-1] for gene in index_file_lines]
    

    
    
    sample_files = glob.glob("./genes/*")
    
    for sample_file in sample_files:
        
        samples = []
        y_values = []    
        
        print("#############################################################")
        print(f"Resuts for {sample_file}:")
        print("#############################################################")
        sample_file = open(sample_file)
    
        for line in sample_file:
        
            split = line.split("\t")
            samples.append(line.split("\t")[0])
            y_values.append(int(split[1][:-1]))
        
        sample_file.close()
    
 
    
        sample_indeces = np.array([gene_list.index(x) for x in samples])
        shuffle_indices = np.arange(sample_indeces.shape[0])
        np.random.shuffle(shuffle_indices)
        sample_indeces = sample_indeces[shuffle_indices]
        y_values = np.array(y_values)[shuffle_indices]
        
        

        ################### get best parameter combination #############################

        
        best_result, best_parameter = GridSearch(["NSPDK", "RLK", "MEDK", "RBF"], [0.1,5,10], adjacency_matrix, sample_indeces, y_values)

        
    
    return


	
if __name__=="__main__":


    main()


    
