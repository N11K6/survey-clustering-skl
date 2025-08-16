#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for dimensionality reduction

@author: nk
"""
import os
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model

def perform_pca(config, dataset):
    # Standardize the data
    sscaler = StandardScaler()
    X_scaled = sscaler.fit_transform(dataset)
    # PCA
    pca = PCA()
    X_pca = pca.fit_transform(X_scaled)
    # Calculate cumulative explained variance
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    # Read desired variance (normalized) for PCA from config
    desired_variance = config['reduction']['pca_variance_threshold']
    print(f'Keeping components with cumulative variance up to {desired_variance}')
    optimal_n_components = len(np.argwhere(cumulative_variance <= desired_variance))
    pca_optimal = PCA(n_components=optimal_n_components)
    print(f'No. of optimal components: {optimal_n_components}')
    X_pca = pca_optimal.fit_transform(X_scaled)
    
    return X_pca

def perform_encoding(config, dataset) -> np.ndarray:
    '''
    Use trained autoencoder to fill a survey dataset with synthesized data.
    '''
    if config['feature_selection']['variance_check']==1 \
        or config['feature_selection']['correlation_check']==1:
            warnings.warn("WARNING: Encoder is selected for dimensionality reduction, and Variance/Correlation check is active.\
                          This may result in incompatible input dimensions!")

    model_name = config['reduction']['model_name']
    # Load the saved autoencoder
    model_path = os.path.join('models', f'{model_name}.keras')
    loaded_encoder = load_model(model_path)
    # Synthesize the missing data
    print('Using encoder for dimensionality reduction.')
    data_reduced = loaded_encoder.predict(dataset)
    scaler = StandardScaler()
    data_reduced_scaled = scaler.fit_transform(data_reduced)

    return data_reduced_scaled

def reduce(config, dataset) -> np.ndarray:
    '''
    Main process for dimensionality reduction
    '''
    if config['reduction']['approach'].lower() == 'pca':
        print('Performing dimensionality reduction using PCA')
        data_reduced = perform_pca(config, dataset)
    elif config['reduction']['approach'].lower() == 'encoder':
        print('Performing Dimensionality reduction using Encoder')
        data_reduced = perform_encoding(config, dataset)
    else:
        raise ValueError('Dimensionality reduction approach can only be: pca, encoder')
    
    return data_reduced
