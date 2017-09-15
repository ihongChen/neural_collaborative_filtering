# -*- coding: utf-8 -*-
"""
Created on Tue Aug 15 16:16:09 2017

@author: 116952
"""
import numpy as np
from time import time
from Dataset import Dataset
from evaluate import evaluate_model
from keras.optimizers import RMSprop, Adam, Adagrad
from keras.regularizers import l2
from keras.models import Sequential, Model
from keras.layers import concatenate
from keras.layers import Input, Embedding, Flatten, merge, Dense
from scipy.sparse import csr_matrix, dok_matrix
import heapq

data = Dataset('./data/ml-1m')

trainMatrix, testRatings, testNegatives = data.trainMatrix, data.testRatings, data.testNegatives
num_users, num_items = data.num_users, data.num_items



# %% 
def get_model(num_users, num_items, layers = [20,10], reg_layers=[0,0]):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0]//2, name = 'user_embedding',
                                  embeddings_initializer= 'uniform', embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0]//2, name = 'item_embedding',
                                  embeddings_initializer= 'uniform', embeddings_regularizer = l2(reg_layers[0]), input_length=1)   
    
    # Crucial to flatten an embedding vector!
    user_latent = Flatten()(MLP_Embedding_User(user_input))
    item_latent = Flatten()(MLP_Embedding_Item(item_input))
    
    # The 0-th layer is the concatenation of embedding layers
    vector = concatenate([user_latent, item_latent])
    
    # MLP layers
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], kernel_regularizer= l2(reg_layers[idx]),
                      activation='relu', 
                      name = 'layer%d' %idx)
        vector = layer(vector)
        
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid', 
                       kernel_initializer='lecun_uniform',
                       name = 'prediction')(vector)
    
    model = Model(inputs=[user_input, item_input], 
                  outputs=prediction)
    
    return model

# %%


def get_train_instances(train, num_negatives):
    
#    if not isinstance(train,dok_matrix):
#        raise ValueError('given matrix must be of dok format')
        
    user_input, item_input, labels = [],[],[]
    num_users,num_items  = train.shape
    for u in range(num_users):
#    for (u, i) in train.keys():
        # positive instance
        i = train[u,].nonzero()[1]
        u = (len(i) + num_negatives)*[u]
        label = len(i)*[1]
        user_input.extend(u)
        item_input.extend(i)
        labels.extend(label)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while j in i:
                j = np.random.randint(num_items)
        
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

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

def getTestNegative(data,num_neg=100):
    user_nums, item_nums = data.shape
    items = np.arange(item_nums)
    
    testNegative = []
#    data_copy = data.copy()
    for user_idx in range(user_nums):
        items_used = data[user_idx,].nonzero()[1]
        items_cand = np.random.choice(items,replace=False,size=200)
        items_neg = [e for e in items_cand if e not in items_used]
        items_neg = items_neg[:num_neg]
        testNegative.append(items_neg)
    return testNegative

def evaluate_recall(model,testRatings,testNegatives,topK=10):
    testNeg_arr = np.array(testNegatives) # numpy negative array 
    testPos_arr = np.array(testRatings)
    test_u = testPos_arr[:,0]
    test_i = testPos_arr[:,1]
    
    (num_user, num_neg_item) = testNeg_arr.shape
    testResult = np.zeros((num_user,num_neg_item + 1))
    testResult[:,-1] = 1
    
    
    test_i.shape = (num_user,1)
    testItem_arr = np.hstack((testNeg_arr,test_i))
        
    testUser_arr = np.repeat(test_u,num_neg_item+1) 
    testUser_arr.shape = (num_user , num_neg_item+1)
    score = 0
    
    for idx in range(num_user):        
        map_item_pred = {}
        pred_val = model.predict([testUser_arr[idx],testItem_arr[idx]])
        for iidx ,item in enumerate(testItem_arr[idx]):
            map_item_pred[item] = pred_val[iidx]
        pred_topK = heapq.nlargest(topK,map_item_pred,key=map_item_pred.get)
        if testItem_arr[idx,-1] in pred_topK:
            score += 1
    return score/num_user

def evaluate_recall2(model,testRatings,trainData,topK=10):
    assert type(trainData) == dok_matrix,'trainData should be dok_matrix'
    
    num_user, num_item = trainData.shape
    testPos_arr = np.array(testRatings)
    test_i = testPos_arr[:,1]
    
    score = 0

    for user in range(num_user):        
        t0 = time()
        map_item_pred = {}
        hist_items = trainData[user,].nonzero()[1]
        
#        testItems = np.array([item for item in range(num_item) if item not in hist_items])
        # use mask to screen item not in hist_item
        testItems = np.arange(num_item) 
        testItem_ma = np.ma.array(testItems,mask=False)
        testItem_ma.mask[hist_items] = True
        testItems = testItem_ma.compressed()
        # users input         
        user_arr = np.full(testItems.shape[0],user,dtype='int32')
        # recommendation list(topK) for single user
        pred_val = model.predict([user_arr, testItems])
        for iidx ,item in enumerate(testItems):
            map_item_pred[item] = pred_val[iidx]
        pred_topK = heapq.nlargest(topK,map_item_pred,key=map_item_pred.get)
        if test_i[user] in pred_topK:
            score += 1
            print('score:{} ,user:{}'.format(score,user))
        print('time for one user:{}(s)'.format(time()-t0))
    return score/num_user

# %% test paper's data
np.random.seed(1)
epochs = 20;
batch_size = 256;
topK = 10;
evaluation_threads = 1;
verbose = 1;
num_negatives = 100;
learning_rate = 0.001
layers = [64,32,16,8]
reg_layers = [0,0,0,0]
model = get_model(num_users = num_users,
                  num_items = num_items, 
                  layers= layers,
                  reg_layers= reg_layers)

#model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')


t1 = time()
#(hits, ndcgs) = evaluate_model(model,testRatings, testNegatives, topK, evaluation_threads)
recall = evaluate_recall(model,testRatings,testNegatives,topK)
#(hits, ndcgs) = evaluate_model(model, testR, testNeg, topK, evaluation_threads)
#hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
print('Init: Recall = %.4f [%.1f s]' %(recall,time()-t1))
#print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))

# Train model
#best_hr, best_ndcg, best_iter = hr, ndcg, -1
best_recall, best_iter = recall,-1
for epoch in range(epochs):
    t1 = time()
    # Generate training instances
    user_input, item_input, labels = get_train_instances(trainMatrix, num_negatives)

    # Training        
    hist = model.fit([np.array(user_input), np.array(item_input)], #input
                     np.array(labels), # labels 
                     batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
    t2 = time()
    
    
    if epoch % verbose == 0:
        (hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
#        recall = evaluate_recall(model,testRatings,testNegatives,topK)
        hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
        loss = hist.history['loss'][0]
        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
              % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
#        print('Iteration %d [%.1f s]: Recall = %.4f, loss = %.4f [%.1f s]' \
#              %(epoch, t2-t1, recall,loss, time()-t2))
#        if recall > best_recall:
#            best_recall, best_iter = recall, epoch


#print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))


# %%%  test bsp data
#
#
#from scipy.io import mmread    
## import data # 
#rb_t = mmread('./data/fund_use.txt') ## read sparse 
#rb_coo = rb_t.transpose()
#rb_csr = rb_coo.tocsr()
#rb_dok = rb_csr.todok()
##rb_csr[1,]
#testR,rb_train = getTestRating(rb_csr)
#testNeg = getTestNegative(rb_csr,num_neg=100)
#
#num_users,num_items = rb_csr.shape
#
#epochs = 20;
#batch_size = 256;
#topK = 10;
#evaluation_threads = 1;
#verbose = 1;
#num_negatives = 4;
#learning_rate = 0.001
#layers = [64,32,16,8]
#reg_layers = [0,0,0,0]
#model = get_model(num_users = num_users,
#                  num_items = num_items, 
#                  layers= layers,
#                  reg_layers= reg_layers)
#
##model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
#model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')
#
#
#t1 = time()
#(hits, ndcgs) = evaluate_model(model,testR, testNeg, topK, evaluation_threads)
#
##(hits, ndcgs) = evaluate_model(model, testR, testNeg, topK, evaluation_threads)
#hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()
#
#print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))
#
## Train model
#best_hr, best_ndcg, best_iter = hr, ndcg, -1
#for epoch in range(epochs):
#    t1 = time()
#    # Generate training instances
#    user_input, item_input, labels = get_train_instances(rb_train, num_negatives)
#
#    # Training        
#    hist = model.fit([np.array(user_input), np.array(item_input)], #input
#                     np.array(labels), # labels 
#                     batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
#    t2 = time()
#    
#    
#    if epoch % verbose == 0:
##        (hits, ndcgs) = evaluate_model(model, testR, testNeg, topK, evaluation_threads)
##        hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
##        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
##              % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
##        if hr > best_hr:
##            best_hr, best_ndcg, best_iter = hr, ndcg, epoch


