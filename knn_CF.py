# -*- coding: utf-8 -*-
"""
Created on Fri Aug 18 14:03:18 2017

@author: 116952
"""
import scipy.sparse as sp
import numpy as np 
import math
from scipy.sparse import csr_matrix
from sklearn.preprocessing import normalize
from itertools import compress


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
    ab = ab.astype('float32')
    # for rows
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
        
def ratings(sim,ui_trans,topn,nn=100,method='user'):
    ''' '''
    sim_topn = knn(sim,nn)
    
    if method == 'user':
        r_mat = sim_topn.dot(ui_trans)
    elif method == 'item': 
        r_mat = ui_trans.dot(sim_topn) 
                
    r_mat = r_mat.tolil()
    rows, cols = ui_trans.nonzero()
    r_mat[rows,cols] = 0 # exclude purchased ratings

    r_mat = r_mat.tocsr()
    r_mat = knn(r_mat,topn)
    r_norm = normalize(r_mat.astype('float64'),norm='l1',axis=1)
    r_norm = r_norm.tocsr()
    return r_norm
    
    


    
# %%
def popularity_guess(ui_trans,topn=10,popular_n=500):
    ## time comsuming
    scores_items = ui_trans.astype('int32').sum(axis=0)
    popular_list = scores_items.argsort()[0,-popular_n:].tolist()[0]

    popular_array = np.zeros((ui_trans.shape[0],topn),dtype=int)
    
    for user in range(ui_trans.shape[0]):
        nz_idx = ui_trans[user,].nonzero()[1]
        bool_pop = [e not in nz_idx for e in popular_list]        
        popular_array[user,] = list(compress(popular_list,bool_pop))[-topn:]
#        print(user)
    return popular_array
    

# %% 
def eval_one_rating(rating_csrMatrix,testRatings,testNegatives,idx):
    _K = len(rating_csrMatrix[0,].data) # topK reccomendation 
    rating = testRatings[idx]
    items = testNegatives[idx] 
    u = rating[0]
    gtItem = rating[1]
    items.append(gtItem)
    # Get prediction scores
    map_item_score = {}
    # users = np.full(len(items), u, dtype = 'int32')
    predictions = np.zeros(len(items),dtype='int32')
    for ii,e in enumerate(items):
        predictions[ii] = (e in rating_csrMatrix[idx,].nonzero()[1])
        
    for i in range(len(items)):
        item = items[i]
        map_item_score[item] = predictions[i]
    items.pop()
    
    # Evaluate top rank list
    ranklist = []
    for k_item_idx,v in map_item_score.items():
        if v>0:
            ranklist.append(k_item_idx)
#    ranklist = heapq.nlargest(_K, map_item_score, key=map_item_score.get)
    hr = getHitRatio(ranklist, gtItem)
    ndcg = getNDCG(ranklist, gtItem)
    return (hr, ndcg)

def getHitRatio(ranklist, gtItem):
    for item in ranklist:
        if item == gtItem:
            return 1
    return 0

def getNDCG(ranklist, gtItem):
    for i in range(len(ranklist)):
        item = ranklist[i]
        if item == gtItem:
            return math.log(2) / math.log(i+2)
    return 0

def getTestRating(data):
    users = data.shape[0]
    testRating = []
    data_copy = data.copy()
    for user_idx in range(users):
        items_cand = data[user_idx,].nonzero()[1]
        testItem = np.random.choice(items_cand)
        testRating.append([user_idx,testItem])
        # remove selected test data        
        data_copy[user_idx,testItem] = 0
        
    data_copy = data_copy.astype('int32')
    
    return testRating,data_copy

    

# %%
#r = ratings(sim,mat,20)
from Dataset import Dataset
data = Dataset('./data/ml-1m')
trainMatrix, testRatings, testNegatives = data.trainMatrix, data.testRatings, data.testNegatives
# %%
sim = jaccard_similarities(trainMatrix.tocsr())
r_user = ratings(sim,trainMatrix.astype('int32'),topn = 10, nn = 100, method = 'user')
# %%
sim_item = jaccard_similarities(trainMatrix.transpose().tocsr())
r_item = ratings(sim_item,trainMatrix.astype('int32'),topn=10,nn=100,method='item')
# %%
pop_a = popularity_guess(trainMatrix,topn=10,popular_n=500)
# %%
score_ubcf = 0; score_ibcf = 0;score_pop = 0
for index,v in testRatings:
    if (v in r_user[index,].nonzero()[1]):
        score_ubcf += 1
    if (v in r_item[index,].nonzero()[1]):
        score_ibcf += 1        
    if (v in pop_a[index,]):
        score_pop += 1
        
recall_ubcf = score_ubcf/len(testRatings) * 100 # 8.7 %
recall_ibcf = score_ibcf/len(testRatings) * 100 # 7.1 %
recall_pop = score_pop /len(testRatings) * 100  # 4.1 %
print("ubcf recall: {0:.1f}% ,\nibcf recall: {1:.1f}% ,\npopular recall: {2:.1f}%".format(recall_ubcf,recall_ibcf,recall_pop))


hits_u, ndcgs_u = [],[]
hits_i, ndcgs_i = [],[]
for idx in range(len(testRatings)):
    (hr_u,ndcg_u) = eval_one_rating(r_user,testRatings,testNegatives,idx)
    (hr_i,ndcg_i) = eval_one_rating(r_item,testRatings,testNegatives,idx)
    hits_u.append(hr_u)
    hits_i.append(hr_i)
    ndcgs_i.append(ndcg_i)
    ndcgs_u.append(ndcg_u)      ## ubcf - 8.7% , ibcf - 7.1 % .... too bad ...algo wrong?,

hits_u_arr = np.array(hits_u)*100
hits_i_arr = np.array(hits_i)*100
print('HR_ubcf :{0:.1f}%,\nHR_ibcf: {1:.1f}%'
      .format(hits_u_arr.mean(),hits_i_arr.mean()))


# %%
from scipy.io import mmread
#
np.random.seed(1)
rb_t = mmread('./data/fund_use.txt') ## read sparse 
rb_coo = rb_t.transpose()
rb_csr = rb_coo.tocsr()
#rb_csr[1,]
testR,rb_train = getTestRating(rb_csr)

pop_rb = popularity_guess(rb_train,topn=10,popular_n=200)

## ubcf 
sim_u = jaccard_similarities(rb_train)

r_user = ratings(sim_u,rb_train.astype('int32'),
                 topn = 10, nn = 200, method = 'user')

# ibcf 
sim_i = jaccard_similarities(rb_train.transpose())
r_item = ratings(sim_i,rb_train.astype('int32'),
                 topn = 10, nn = 200, method = 'item')
score_ubcf = 0; score_ibcf = 0;score_pop = 0
for index,v in testR:
    if (v in r_user[index,].nonzero()[1]):
        score_ubcf += 1
    if (v in r_item[index,].nonzero()[1]):
        score_ibcf += 1        
    if (v in pop_rb[index,]):
        score_pop += 1

recall_pop = score_pop/len(testR)*100
recall_ubcf = score_ubcf / len(testR) * 100
recall_ibcf = score_ibcf / len(testR) * 100
print("ubcf recall : {:.1f}%".format(recall_ubcf)) # 25.0 %
print("ibcf recall: {:.1f}%".format(recall_ibcf)) #22.1 %
print("popular recall: {:.1f}%".format(recall_pop)) # 20.3%


