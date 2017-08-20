# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:03:18 2017

@author: 116952
"""
import scipy.sparse as sp
import numpy as np 
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize

#A = csr_matrix([[4,1, 2, 0], [0,0, 0, 3], [0,4, 0, 5]])
#A.transpose().dot(A)

def jaccard_similarities(mat):
    '''
    given a sparse matrix, calculate jaccard sim 
    == 
    input : u-i mat, 
    output : u-u jaccard similarity sparse matrix
    
    ** ref : http://na-o-ys.github.io/others/2015-11-07-sparse-vector-similarities.html
    '''
    rows_sum = mat.getnnz(axis=1)  # 
    ab = mat.dot(mat.T) # mat x t(mat)

    # for rows= to just = for your 2 sparse arrays:
    aa = np.repeat(rows_sum, ab.getnnz(axis=1))
    # for columns
    bb = rows_sum[ab.indices]

    similarities = ab.copy()
    similarities.data /= (aa + bb - ab.data)
    similarities.setdiag(0)

    return similarities

# %%

def max_n(row_data, row_indices, n):
    i = row_data.argsort()[-n:]
    top_values = row_data[i]
    top_indices = row_indices[i]  # do the sparse indices matter?
    return top_values, top_indices, i    

def knn(sim,n):
    sim_topn = sim.tolil()
    for i in range(sim_topn.shape[0]):
        d,r=max_n(np.array(sim_topn.data[i]),
               np.array(sim_topn.rows[i]),n)[:2]
        sim_topn.data[i] = d.tolist()
        sim_topn.rows[i] = r.tolist()
    return sim_topn
        
def ratings(sim,ui_trans,topn):
    sim_topn = knn(sim,topn)
    r_mat = sim_topn.dot(ui_trans)
#    r_norm = r_mat/r_mat.sum(axis=1)
    r_norm = normalize(r_mat, norm='l1', axis=1)
    r_norm = r_norm.tolil()
    rows, cols = ui_trans.nonzero()
    r_norm[rows,cols] = 0 # exclude purchased ratings
#    r_norm[ui_trans.indices] = 0  # wrong
    r_norm = r_norm.tocsr()
    return r_norm
    
    

    
#arr = np.array([[0,5,3,0,2],[6,0,4,9,0],[0,0,0,6,8]])
#arr_sp = sp.csc_matrix(arr)
#arr_ll = arr_sp.tolil()
#for i in range(arr_ll.shape[0]):
#     d,r=max_n(np.array(arr_ll.data[i]),
#               np.array(arr_ll.rows[i]),2)[:2]
#     arr_ll.data[i] = d.tolist()
#     arr_ll.rows[i] = r.tolist()


    
# %%

mat = sp.rand(3000, 1000, 0.01, format='csr')
#mat_lil = mat.tolil()
mat.data[:] = 1 # binarize
#mat.toarray()
sim = jaccard_similarities(mat)

# %%
r = ratings(sim,mat,20)
# %%
