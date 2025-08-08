#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for dimensionality reduction

@author: nk
"""
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

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

