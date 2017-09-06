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


# %% test paper's data
epochs = 20;
batch_size = 256;
topK = 10;
evaluation_threads = 1;
verbose = 1;
num_negatives = 4;
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
(hits, ndcgs) = evaluate_model(model,testRatings, testNegatives, topK, evaluation_threads)

#(hits, ndcgs) = evaluate_model(model, testR, testNeg, topK, evaluation_threads)
hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()

print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))

# Train model
best_hr, best_ndcg, best_iter = hr, ndcg, -1
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
        hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
              % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch


print("End. Best Iteration %d:  HR = %.4f, NDCG = %.4f. " %(best_iter, best_hr, best_ndcg))


# %%%  test bsp data


from scipy.io import mmread    
# import data # 
rb_t = mmread('./data/fund_use.txt') ## read sparse 
rb_coo = rb_t.transpose()
rb_csr = rb_coo.tocsr()
rb_dok = rb_csr.todok()
#rb_csr[1,]
testR,rb_train = getTestRating(rb_csr)
testNeg = getTestNegative(rb_csr,num_neg=100)

num_users,num_items = rb_csr.shape

epochs = 20;
batch_size = 256;
topK = 10;
evaluation_threads = 1;
verbose = 1;
num_negatives = 4;
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
(hits, ndcgs) = evaluate_model(model,testR, testNeg, topK, evaluation_threads)

#(hits, ndcgs) = evaluate_model(model, testR, testNeg, topK, evaluation_threads)
hr, ndcg = np.array(hits).mean(), np.array(ndcgs).mean()

print('Init: HR = %.4f, NDCG = %.4f [%.1f]' %(hr, ndcg, time()-t1))

# Train model
best_hr, best_ndcg, best_iter = hr, ndcg, -1
for epoch in range(epochs):
    t1 = time()
    # Generate training instances
    user_input, item_input, labels = get_train_instances(rb_train, num_negatives)

    # Training        
    hist = model.fit([np.array(user_input), np.array(item_input)], #input
                     np.array(labels), # labels 
                     batch_size=batch_size, epochs=1, verbose=0, shuffle=True)
    t2 = time()
    
    
    if epoch % verbose == 0:
        (hits, ndcgs) = evaluate_model(model, testR, testNeg, topK, evaluation_threads)
        hr, ndcg, loss = np.array(hits).mean(), np.array(ndcgs).mean(), hist.history['loss'][0]
        print('Iteration %d [%.1f s]: HR = %.4f, NDCG = %.4f, loss = %.4f [%.1f s]' 
              % (epoch,  t2-t1, hr, ndcg, loss, time()-t2))
        if hr > best_hr:
            best_hr, best_ndcg, best_iter = hr, ndcg, epoch


