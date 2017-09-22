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
from keras.layers import concatenate, multiply
from keras.layers import Input, Embedding, Flatten, merge, Dense
from scipy.sparse import csr_matrix, dok_matrix
import heapq


# %% build MLP model 
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

# %% build NeuMF model (neural matrix factorize)
def get_NeuMF_model(num_users, num_items, mf_dim=10, layers=[10], reg_layers=[0], reg_mf=0):
    assert len(layers) == len(reg_layers)
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, output_dim = mf_dim, name = 'mf_embedding_user',
                                  embeddings_initializer = 'uniform', embeddings_regularizer = l2(reg_mf), input_length=1)
    MF_Embedding_Item = Embedding(input_dim = num_items, output_dim = mf_dim, name = 'mf_embedding_item',
                                  embeddings_initializer = 'uniform', embeddings_regularizer = l2(reg_mf), input_length=1)   

    MLP_Embedding_User = Embedding(input_dim = num_users, output_dim = layers[0]//2, name = "mlp_embedding_user",
                                  embeddings_initializer = 'uniform', embeddings_regularizer = l2(reg_layers[0]), input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, output_dim = layers[0]//2, name = 'mlp_embedding_item',
                                  embeddings_initializer = 'uniform', embeddings_regularizer = l2(reg_layers[0]), input_length=1)   
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))
#    mf_vector = merge([mf_user_latent, mf_item_latent], mode = 'mul') # element-wise multiply
    mf_vector = multiply([mf_user_latent, mf_item_latent])
    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
#    mlp_vector = merge([mlp_user_latent, mlp_item_latent], mode = 'concat')
    mlp_vector = concatenate([mlp_user_latent, mlp_item_latent])
    for idx in range(1, num_layer):
        layer = Dense(layers[idx], 
                      kernel_regularizer= l2(reg_layers[idx]), 
                      activation='relu', 
                      name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
#    predict_vector = merge([mf_vector, mlp_vector], mode = 'concat')
    predict_vector = concatenate([mf_vector, mlp_vector])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid',
                       kernel_initializer='lecun_uniform',
                       name = "prediction")(predict_vector)
    
    model = Model(inputs=[user_input, item_input], 
                  outputs=prediction)
    
    return model

# %%
    
def get_NeuMF_features_model(num_users,                              
                             num_items, 
                             num_users_features,
                             num_items_features,
                             mf_dim=10, layers=[10],                             
                             reg_layers=[0], reg_mf=0):
    ''' '''
    assert len(layers) == len(reg_layers)
    # modified MLP layers with num_features (user + items)
    layers[0] = layers[0] + num_users_features + num_items_features
    
    #
    num_layer = len(layers) #Number of layers in the MLP
    # Input variables
    user_input = Input(shape=(1,), dtype='int32', name = 'user_input')
    item_input = Input(shape=(1,), dtype='int32', name = 'item_input')
    # Features variables 
    user_features = Input(shape=(num_users_features,), dtype='float32',name ='user_features')
    item_features = Input(shape=(num_items_features,), dtype='float32',name = 'item_features')
#    user_feature = Input(shape=)
    # Embedding layer
    MF_Embedding_User = Embedding(input_dim = num_users, 
                                  output_dim = mf_dim, 
                                  name = 'mf_embedding_user',
                                  embeddings_initializer = 'uniform', 
                                  embeddings_regularizer = l2(reg_mf), 
                                  input_length=1)
    
    MF_Embedding_Item = Embedding(input_dim = num_items, 
                                  output_dim = mf_dim, 
                                  name = 'mf_embedding_item',
                                  embeddings_initializer = 'uniform', 
                                  embeddings_regularizer = l2(reg_mf), 
                                  input_length=1)   

    MLP_Embedding_User = Embedding(input_dim = num_users, 
                                   output_dim = layers[0]//2, 
                                   name = "mlp_embedding_user",
                                   embeddings_initializer = 'uniform', 
                                   embeddings_regularizer = l2(reg_layers[0]), 
                                   input_length=1)
    MLP_Embedding_Item = Embedding(input_dim = num_items, 
                                   output_dim = layers[0]//2, 
                                   name = 'mlp_embedding_item',
                                   embeddings_initializer = 'uniform', 
                                   embeddings_regularizer = l2(reg_layers[0]), 
                                   input_length=1)   
    
    # MF part
    mf_user_latent = Flatten()(MF_Embedding_User(user_input))
    mf_item_latent = Flatten()(MF_Embedding_Item(item_input))

    mf_vector = multiply([mf_user_latent, mf_item_latent]) # element-wise multiply
    # MLP part 
    mlp_user_latent = Flatten()(MLP_Embedding_User(user_input))
    mlp_item_latent = Flatten()(MLP_Embedding_Item(item_input))
    mlp_vector = concatenate([mlp_user_latent,mlp_item_latent])

    for idx in range(1, num_layer):
        layer = Dense(layers[idx], 
                      kernel_regularizer= l2(reg_layers[idx]), 
                      activation='relu', 
                      name="layer%d" %idx)
        mlp_vector = layer(mlp_vector)

    # Concatenate MF and MLP parts
    #mf_vector = Lambda(lambda x: x * alpha)(mf_vector)
    #mlp_vector = Lambda(lambda x : x * (1-alpha))(mlp_vector)
#    predict_vector = merge([mf_vector, mlp_vector], mode = 'concat')
    predict_vector = concatenate([mf_vector,mlp_vector,user_features,item_features])
    
    # Final prediction layer
    prediction = Dense(1, activation='sigmoid',
                       kernel_initializer='lecun_uniform',
                       name = "prediction")(predict_vector)
    
    model = Model(inputs=[user_input,user_features, item_input,item_features], 
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
    ''' recall w.r.t Negative samplings '''
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


def evaluate_recall_features(model,testRatings,testNegatives,
                             user_features,item_features,
                             topK=10):
    ''' recall w.r.t Negative samplings and features '''
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
    
    if type(user_features)==int and user_features ==0:
        user_features = np.array([]); 
        user_features.shape = (num_user,0)
    if type(item_features)==int and item_features == 0:
        item_features =np.array([])
        item_features.shape = (num_user,0)
        
    for idx in range(num_user):        
        map_item_pred = {}
        item_features_idx = np.array([item_features[idx]] * (num_neg_item+1))
        user_features_idx = np.array([user_features[idx]] * (num_neg_item+1))
        pred_val = model.predict([testUser_arr[idx],
                                  user_features_idx,
                                  testItem_arr[idx],
                                  item_features_idx
                                  ])
        for iidx ,item in enumerate(testItem_arr[idx]):
            map_item_pred[item] = pred_val[iidx]
        pred_topK = heapq.nlargest(topK,map_item_pred,key=map_item_pred.get)
        if testItem_arr[idx,-1] in pred_topK:
            score += 1
    return score/num_user

def evaluate_recall2(model,testRatings,trainData,topK=10):
    '''recall w.r.t all datasets (not sampling) '''
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
# =============================================================================
# 測試論文 Movie Lens datasets 
# =============================================================================
    
data = Dataset('./data/ml-1m')

trainMatrix, testRatings, testNegatives = data.trainMatrix, data.testRatings, data.testNegatives
num_users, num_items = data.num_users, data.num_items

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
# =============================================================================
# MLP
#
#model = get_model(num_users = num_users,
#                  num_items = num_items, 
#                  layers= layers,
#                  reg_layers= reg_layers)
# =============================================================================
# NeuMF


model = get_NeuMF_model(num_users=num_users,
                        num_items=num_items,
                        layers=layers,
                        reg_layers=reg_layers,
                        reg_mf=0)

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
# =============================================================================
# Bank SinoPac data --- 基金推薦問題
# =============================================================================

from scipy.io import mmread    
import pandas as pd 
# import data # 
rb_t = mmread('./data/fund_use.txt') ## read sparse matrix , transaction 
rb_coo = rb_t.transpose()
rb_csr = rb_coo.tocsr()
rb_dok = rb_csr.todok() 

# load features data (users +items )
features_df = pd.read_csv('./data/features_used.csv',sep=',',dtype='int32')
features_arr = features_df.values

#rb_csr[1,]
testR,rb_train = getTestRating(rb_csr)
testNeg = getTestNegative(rb_csr,num_neg=100)

num_users,num_items = rb_csr.shape

epochs = 20;
batch_size = 256;
topK = 10;
evaluation_threads = 1;
verbose = 1;
num_negatives = 50;
learning_rate = 0.001
layers = [64,32,16,8]
#reg_layers = [0.1,0.5,0.2,0.3]
reg_layers = [0,0,0,0]
#model = get_NeuMF_model(num_users = num_users,
#                  num_items = num_items, 
#                  layers= layers,
#                  reg_layers= reg_layers) ## 愈train 愈爛!!! 
model = get_model(num_users,num_items,layers, reg_layers)
#
#model = get_NeuMF_features_model(num_users,num_items,num_users_features=0,
#                                 num_items_features=0,mf_dim=10,layers=layers,
#                                 reg_layers = reg_layers, reg_mf = 0)
#model.compile(optimizer=RMSprop(lr=learning_rate), loss='binary_crossentropy')
model.compile(optimizer=Adam(lr=learning_rate), loss='binary_crossentropy')


t1 = time()


#
#recall = evaluate_recall_features(model = model,
#                                  testRatings = testR,
#                                  testNegatives = testNeg,
#                                  user_features = 0,
#                                  item_features = 0,
#                                  topK = topK)

recall = evaluate_recall(model,testR,testNeg,topK)
#
print('Init: Recall = %.4f [%.1f sec]' %(recall, time()-t1))
#
# Train model

best_recall, best_iter = recall,-1
for epoch in range(epochs):
    t1 = time()
    # Generate training instances
    user_input, item_input, labels = get_train_instances(rb_train, num_negatives)
#    user_features = np.array([features_arr[user] for user in user_input])
    user_features = np.array([],dtype='int32')
    user_features.shape = (len(item_input),0)
    item_features = np.array([],dtype='int32')
    item_features.shape = (len(user_input),0)
    # Training        
    
    hist = model.fit(
            [
                np.array(user_input),
                np.array(item_input)
            ],
            np.array(labels),
            batch_size = batch_size,
            epochs = 1,
            verbose = 0,
            shuffle = True
            )
#    hist = model.fit(
#            [
#                np.array(user_input),
#                user_features,
#                np.array(item_input),
#                item_features
#            ], #input
#            np.array(labels), # labels 
#            batch_size=batch_size,
#            epochs=1,
#            verbose=0,
#            shuffle=True)
    t2 = time()
    
    
    if epoch % verbose == 0:
#        recall = evaluate_recall_features(model = model, 
#                                          testRatings = testR,
#                                          testNegatives = testNeg,
#                                          user_features = 0,
#                                          item_features = 0,
#                                          topK = topK)
        recall = evaluate_recall(model = model, 
                                 testRatings = testR,
                                 testNegatives = testNeg,
                                 topK = topK)
        loss = hist.history['loss'][0]
        print('Iteration %d [%.1f s]: Recall = %.4f, loss = %.4f [%.1f s]' 
              % (epoch,  t2-t1, recall, loss, time()-t2))


