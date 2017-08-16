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
    user_input, item_input, labels = [],[],[]
    num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        user_input.append(u)
        item_input.append(i)
        labels.append(1)
        # negative instances
        for t in range(num_negatives):
            j = np.random.randint(num_items)
            while (u,j) in train.keys():
                j = np.random.randint(num_items)
            user_input.append(u)
            item_input.append(j)
            labels.append(0)
    return user_input, item_input, labels

# %%
#if __name__ == '__main__':
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
(hits, ndcgs) = evaluate_model(model, testRatings, testNegatives, topK, evaluation_threads)
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