#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Clustering using HDBSCAN

@author: nk
"""
import numpy as np
from sklearn.cluster import HDBSCAN
from sklearn.metrics import silhouette_score, davies_bouldin_score, calinski_harabasz_score
import matplotlib.pyplot as plt

def perform_HDBSCAN(config, data_reduced):
    min_cluster_size = config['clustering']['min_cluster_size']
    data_clustered = HDBSCAN(min_cluster_size=min_cluster_size).fit_predict(data_reduced)
    return data_clustered

def visualize_clusters(data_reduced, data_clustered):
    # Plot results in 2 Dimensions
    plt.clf()
    plt.scatter(data_reduced[:, 0], data_reduced[:, 1], c=data_clustered, cmap='plasma', marker='o', s=50)
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title('2D Cluster Visualization')
    plt.colorbar(label='Cluster')
    #plt.show()

    return plt

def perform_clustering(config, data_reduced):
    # Perform clustering
    print('Performing clustering using HDBSCAN.')
    data_clustered = perform_HDBSCAN(config, data_reduced)
    
    # Plot clustered data
    cluster_plot = visualize_clusters(data_reduced, data_clustered)
    
    # --- Error Handling for All-Noise Clustering ---
    unique_labels = np.unique(data_clustered)
    n_clusters = len(unique_labels)
    # Check if all points are labeled as noise (-1)
    # This happens if n_clusters is 1 and the only label is -1
    if n_clusters < 2 or (n_clusters == 1 and unique_labels[0] == -1):
        print(f"Warning: HDBSCAN found no clusters. All {len(data_clustered)} points labeled as noise (-1).")
        # Assign NaN to metrics as they cannot be calculated
        sil_score = np.nan
        db_score = np.nan
        ch_score = np.nan
    else:
        # Calculate clustering evaluation metrics
        sil_score = silhouette_score(data_reduced, data_clustered)
        print(f"Silhouette Score: {sil_score}")
        db_score = davies_bouldin_score(data_reduced, data_clustered)
        print(f"Davies-Bouldin Index: {db_score}")
        ch_score = calinski_harabasz_score(data_reduced, data_clustered)
        print(f"Calinski-Harabasz Index: {ch_score}")
        
    metrics_dict = {}
    metrics_dict['sil_score'] = sil_score
    metrics_dict['db_score'] = db_score
    metrics_dict['ch_score'] = ch_score

    return data_clustered, metrics_dict, cluster_plot
