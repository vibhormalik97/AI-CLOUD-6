import boto3
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import GlobalMaxPool1D
from tensorflow.keras.layers import MaxPooling1D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Embedding
from tensorflow.keras.layers import Conv1D


def keras_model_fn(_, config):
    """
    Creating a CNN model for sentiment modeling
    """
    # load the whole embedding into memory from s3
    embeddings_index = dict()
    s3 = boto3.resource('s3')
    obj = s3.Object('aiops-assignment6', 'glove.txt')
    body = obj.get()['Body'].read().decode("utf-8").splitlines()
    
    for line in body:
        values = line.split()
        word = values[0]
        coefs = np.array(values[1:], dtype='float32')
        embeddings_index[word] = coefs

    print('Loaded %s word vectors.' % len(embeddings_index))

    vocab_size = config["embeddings_dictionary_size"]

    embedding_matrix = np.zeros((len(embeddings_index.keys()), en(embeddings_index['the'])))
    for index, key in zip(range(0, n), embeddings_index.keys()):
        embedding_matrix[index] = embeddings_index[key]

    # define model
    cnn_model = Sequential()
    cnn_model.add(Embedding(vocab_size, 25, weights=[embedding_matrix], input_length=40, trainable=True, name='embedding'))
    cnn_model.add(Conv1D(filters = 128, kernel_size = 5, strides = 1, padding = 'valid', activation = 'relu'))
    cnn_model.add(MaxPooling1D(3))
    cnn_model.add(GlobalMaxPool1D())
    cnn_model.add(Dense(100, activation = 'relu'))
    cnn_model.add(Dropout(0.2))
    cnn_model.add(Dense(1, activation = 'sigmoid'))
    adam = Adam(lr=0.001)
    cnn_model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    
    print('Defined model')
    return cnn_model

def save_model(model, output):
    """
    Method to save models in SaveModel format with signature to allow for serving
    """
    # Saves file to SageMaker folder
    tf.saved_model.save(model, os.path.join(output, "1"))

    print("Model successfully saved at: {}".format(output))