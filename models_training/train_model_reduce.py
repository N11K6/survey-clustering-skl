#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Encoder for dimensionality reduction.

IMPORTANT:
    Must be trained with the appropriate reduced dataset!
    Take note of what features are included in the dataset 
    the model is trained with!
    
    It is not recommended to perform Variance & Correlation checks
    on the datasets if an Encoder is used since the number of kept
    features might vary depending on the data.

@author: nk
"""
# Import dependencies
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd

def build_train_save_model(config:dict, dataset_full: pd.DataFrame, encoding_dim=2):
    '''
    Build/Train/Save a simple autoencoder (CPU-friendly architecture)
    This will be used for synthesizing missing data
    '''
    input_dim = dataset_full.shape[1]
    # Smaller architecture for CPU training
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(32, activation='relu')(input_layer)
    encoder = Dense(8, activation='relu')(encoder)
    encoder = Dense(encoding_dim, activation='relu')(encoder)  # Bottleneck
    decoder = Dense(8, activation='relu')(encoder)
    decoder = Dense(32, activation='relu')(decoder)
    decoder = Dense(input_dim, activation='sigmoid')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    '''
    Train the model
    '''
    print('Training Autoencoder.')
    autoencoder.fit(dataset_full, dataset_full,
                epochs=100,
                batch_size=32,
                shuffle=True,
                validation_split=0.2,
                verbose=1)
    print('Training finished.')
    
    '''
    Save the model
    '''
    # Get model name from config
    model_name = config['reduction']['model_name']
    encoder_model = Model(inputs=input_layer, outputs=encoder)
    model_name = 'model_reduce2'
    model_path = os.path.join('..', 'models', f'{model_name}.keras')
    encoder_model.save(model_path)
    print(f'Encoder saved as {model_name}.keras')

    return 0

if __name__ == "__main__":
    config = {'reduction':{'model_name':"model_synthesize"}}
    dataset_path = os.path.join('..', 'data', 'dataset_full.xlsx')
    dataset = pd.read_excel(dataset_path)
    build_train_save_model(config, dataset)