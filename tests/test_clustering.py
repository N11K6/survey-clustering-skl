#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 15:00:15 2025

@author: nk
"""
import numpy as np
from sklearn.datasets import make_blobs
import matplotlib
matplotlib.use('Agg') # Use non-interactive backend for testing
import matplotlib.pyplot as plt
# Assuming the functions are in 'analysis_pipeline.py'
from pipeline.clustering import perform_HDBSCAN, visualize_clusters, perform_clustering

# --- Test perform_HDBSCAN ---
def test_perform_HDBSCAN():
    """Test HDBSCAN clustering on a simple blob dataset."""
    # Give minimum cluster size through config
    config = {'clustering':{'min_cluster_size':20}}

    # Create a dataset with clear clusters
    X, _ = make_blobs(n_samples=100, centers=3, n_features=2, random_state=42)
    
    labels = perform_HDBSCAN(config, X)

    assert isinstance(labels, np.ndarray)
    assert labels.shape == (X.shape[0],)
    # Check that some points are clustered (not all -1)
    # HDBSCAN labels noise points as -1
    unique_labels = np.unique(labels)
    assert len(unique_labels) > 1 # Should have at least noise (-1) and one cluster
    # assert -1 in unique_labels # Likely, but depends on data and min_cluster_size


# --- Test visualize_clusters ---
def test_visualize_clusters():
    """Test that visualize_clusters creates a matplotlib figure."""
    # Create simple 2D data
    X = np.random.rand(20, 2)
    labels = np.array([0] * 10 + [1] * 10) # Simple labels

    fig_or_plt = visualize_clusters(X, labels)

    # Check if it returns a plt object (as per current implementation)
    # Or check if it returns a Figure object if modified
    # assert isinstance(fig_or_plt, matplotlib.figure.Figure) # If returning fig
    assert fig_or_plt is plt # Current implementation returns plt

    # Ensure the plot was created
    assert len(plt.get_fignums()) > 0

    # Clean up the plot
    plt.close('all')


# --- Test perform_clustering ---
def test_perform_clustering():
    """Test the full clustering pipeline."""
    # Give minimum cluster size through config
    config = {'clustering':{'min_cluster_size':20}}
    # Create a dataset with clear clusters suitable for HDBSCAN
    X, _ = make_blobs(n_samples=120, centers=3, n_features=4, random_state=42)
    # Reduce to 2D for visualization (or use the reduced data directly if available)
    # For this test, we'll use the full 4D data, assuming HDBSCAN can handle it.
    # If your pipeline expects 2D, adjust accordingly.

    data_clustered, metrics_dict, cluster_plot = perform_clustering(config, X)

    # Assertions for data_clustered
    assert isinstance(data_clustered, np.ndarray)
    assert data_clustered.shape == (X.shape[0],)

    # Assertions for metrics_dict
    assert isinstance(metrics_dict, dict)
    assert 'sil_score' in metrics_dict
    assert 'db_score' in metrics_dict
    assert 'ch_score' in metrics_dict
    assert isinstance(metrics_dict['sil_score'], (int, float))
    assert isinstance(metrics_dict['db_score'], (int, float))
    assert isinstance(metrics_dict['ch_score'], (int, float))

    # Check metric ranges (basic sanity)
    # Silhouette: -1 to 1 (higher is better)
    # Davies-Bouldin: 0 to inf (lower is better)
    # Calinski-Harabasz: 0 to inf (higher is better)
    # These checks depend on successful clustering producing reasonable results.
    # With good clusters, Silhouette should be positive, DB low, CH high.
    # We cannot assert exact values due to stochasticity, but basic bounds.
    sil_score = metrics_dict['sil_score']
    db_score = metrics_dict['db_score']
    ch_score = metrics_dict['ch_score']

    # Silhouette score should be <= 1
    assert sil_score <= 1.0
    # Davies-Bouldin should be >= 0
    assert db_score >= 0.0
    # Calinski-Harabasz should be >= 0
    assert ch_score >= 0.0

    # Assertions for cluster_plot (matplotlib object)
    assert cluster_plot is plt # Based on current implementation
    assert len(plt.get_fignums()) > 0 # A figure should be created

    # Clean up plots
    plt.close('all')


# --- Edge Case Test for perform_clustering ---
def test_perform_clustering_all_noise():
    """Test clustering when HDBSCAN might label everything as noise."""
    # Give minimum cluster size through config
    config = {'clustering':{'min_cluster_size':20}}
    # Create data that's likely to be labeled as noise
    # e.g., very sparse data or data not meeting min_cluster_size
    X = np.random.rand(50, 3) * 0.1 # Very small spread

    # Capture print output if needed to verify behavior
    # from io import StringIO
    # import sys
    # capturedOutput = StringIO()
    # sys.stdout = capturedOutput

    data_clustered, metrics_dict, cluster_plot = perform_clustering(config, X)

    # All points might be labeled as -1 (noise)
    # The metric calculation might fail or produce specific values
    # Silhouette score calculation fails if there's only one cluster (including -1)
    # Davies-Bouldin needs more than one cluster
    # Calinski-Harabasz needs more than one cluster
    # If all points are noise, metrics calculation might raise an error.

    # Check the state of data_clustered
    assert isinstance(data_clustered, np.ndarray)
    assert data_clustered.shape == (X.shape[0],)
    unique_labels = np.unique(data_clustered)
    # Likely all -1, but check if metrics calculation succeeded or failed
    if len(unique_labels) <= 1: # Only noise or one cluster found
        # Metrics calculation should have failed
        # The function currently doesn't handle this, so it might raise an error
        # If it raises, the test should reflect that or the function should be made robust
        # For now, assume the function might raise an error in this case
        # This test documents a potential edge case
        pass # Or assert specific behavior/error if function is updated to handle it


    # Clean up plots
    plt.close('all')

    # Reset stdout if captured
    # sys.stdout = sys.__stdout__
    # print(capturedOutput.getvalue()) # Print captured output for debugging
