#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Auto Encoder for synthesizing data

@author: nk
"""
import os
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd

def build_train_model(dataset_X: pd.DataFrame, dataset_y: pd.DataFrame, model_name: str):
    '''
    Build/Train/Save a simple autoencoder (CPU-friendly architecture)
    This will be used for synthesizing missing data
    '''
    
    # 1. Build the autoencoder
    input_dim = dataset_X.shape[1]
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
    autoencoder.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    
    # 2. Train with fewer epochs and larger batch size (better for CPU)
    print('Training autoencoder for data synthesis.')
    autoencoder.fit(dataset_X, dataset_y,
                    epochs=100,
                    batch_size=32,
                    shuffle=True,
                    validation_split=0.2,
                    verbose=2)
    # 3. Save the model
    model_path = os.path.join('..', 'models', f'{model_name}.keras')
    autoencoder.save(model_path)
        
    # 4. Done
    print('Autoencoder ready.')
    
    return 0
    
def prepare_data(dataset: pd.DataFrame):
    '''
    Split the dataset to:
        X -> Core questions 
        y -> Recontact questions
        train -> Entries where Recontact is known
        missing -> Entries without Recontact, to be synthesized
    '''
    dataset_train = dataset.dropna()
    re_cols_lst = [col for col in dataset.columns if 'core_re' in col]
    dataset_X = dataset_train.drop(re_cols_lst, axis=1).reset_index(drop=True)
    dataset_y = dataset_train[re_cols_lst].reset_index(drop=True)
    
    return dataset_X, dataset_y

def main():
    dataset_path = os.path.join('..', 'data', 'dataset.xlsx')
    dataset = pd.read_excel(dataset_path)
    dataset_X, dataset_y = prepare_data(dataset)
    model_name = 'model_synthesize'
    build_train_model(dataset_X, dataset_y, model_name)
    
    return 0
    
if __name__ == "__main__":
    main()
    