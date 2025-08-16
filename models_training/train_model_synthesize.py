#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Autoencoder used to synthesize missing data

@author: nk
"""
# Import dependencies
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd

def prepare_data(dataset: pd.DataFrame):
    '''
    Split the dataset to:
        X -> Core questions 
        y -> Recontact questions
        train -> Entries where Recontact is known
        missing -> Entries without Recontact, to be synthesized
    '''
    print('Preparing data for model training.')
    dataset_train = dataset.dropna()
    # Keep the Recontact columns to separate from Core columns
    re_cols_lst = [col for col in dataset.columns if 'core_re' in col]
    # Use the Core as X
    dataset_X = dataset_train.drop(re_cols_lst, axis=1).reset_index(drop=True)
    # Use the Recontact as y
    dataset_y = dataset_train[re_cols_lst].reset_index(drop=True)
    
    return dataset_X, dataset_y

def build_model(dataset_X: pd.DataFrame, dataset_y: pd.DataFrame, encoding_dim=2):
    '''
    Build/Train/Save a simple autoencoder (CPU-friendly architecture)
    This will be used for synthesizing missing data
    '''
    # Input should reflect the shape of X
    input_dim = dataset_X.shape[1]
    # Output should reflect the shape of y
    output_dim = dataset_y.shape[1]
    # Compact architecture, suited for the small dataset we have
    input_layer = Input(shape=(input_dim,))
    encoder = Dense(16, activation='relu')(input_layer)
    encoder = Dense(8, activation='relu')(encoder)
    encoder = Dense(4, activation='relu')(encoder)  # Bottleneck
    decoder = Dense(8, activation='relu')(encoder)
    decoder = Dense(16, activation='relu')(decoder)
    decoder = Dense(output_dim, activation='sigmoid')(decoder)
    autoencoder = Model(inputs=input_layer, outputs=decoder)
    # Compile the model
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    return autoencoder
    
def train_model(model, dataset_X: pd.DataFrame, dataset_y: pd.DataFrame):
    '''
    Train the model
    '''
    print('Training Autoencoder model.')
    model.fit(dataset_X, dataset_y,
                    epochs=100,
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.2,
                    verbose=1)
    print('Training finished.')
    
    return model
    
def save_model(config:dict, model):
    '''
    Save the model
    '''
    # Get model name from config
    model_name = config['missing_data']['model_name']
    model_path = os.path.join('..', 'models', f'{model_name}.keras')
    model.save(model_path)
    print(f'Autoencoder saved as {model_name}.keras')

    return 0

def build_train_save(config:dict, dataset:pd.DataFrame):
    dataset_X, dataset_y = prepare_data(dataset)
    model = build_model(dataset_X, dataset_y)
    model = train_model(model, dataset_X, dataset_y)
    save_model(config, model)
    return 0

if __name__ == "__main__":
    config = {'data_source':{'type':'local',
                             'path': "dataset.xlsx"},
              'missing_data':{'model_name':"model_synthesize"}}
    dataset_path = os.path.join('..', 'data', 'dataset.xlsx')
    dataset = pd.read_excel(dataset_path)
    build_train_save(config, dataset)