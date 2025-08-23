#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for dimensionality reduction

@author: nk
"""
import os
import tempfile
import numpy as np
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from tensorflow.keras.models import load_model
from pipeline.data_load import connect_to_s3

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
    Use trained encoder to perform dimensionality reduction.
    '''
    if config['feature_selection']['variance_check']==1 \
        or config['feature_selection']['correlation_check']==1:
            warnings.warn("WARNING: Encoder is selected for dimensionality reduction, and Variance/Correlation check is active.\
                          This may result in incompatible input dimensions!")
    
    def load_trained_model(config):
        '''
        Load the trained model for data synthesis.
        If data source = local, will look for model in the local models folder
        If data source = s3, will look for files in s3 storage specified in config
        '''
        model_name = config['reduction']['model_name']
        if config['data_source']['type'] == 'local':
            model_path = os.path.join('models', f'{model_name}')
            trained_model = load_model(model_path)
        else:
            # Connect to s3
            source = config['data_source']
            s3 = connect_to_s3(source)
            # Parsing path
            path = os.path.join(source['s3_path'],config['reduction']['model_name'])
            bucket, key = path.replace("s3://", "").split("/", 1)
            # Download model file from s3 to memory
            obj = s3.get_object(Bucket=bucket, Key=key)
            # For .keras format, we need to save to temporary file
            # as it's a directory-like structure that can't be loaded directly from memory
            with tempfile.NamedTemporaryFile(delete=False, suffix='.keras') as tmp_file:
                tmp_file.write(obj['Body'].read())
                tmp_file_path = tmp_file.name
            trained_model = load_model(tmp_file_path)
            # Clean up temporary file
            os.unlink(tmp_file_path)

        return trained_model

    # Load the saved encoder
    trained_model = load_trained_model(config)
    # Synthesize the missing data
    print('Using encoder for dimensionality reduction.')
    data_reduced = trained_model.predict(dataset)
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
