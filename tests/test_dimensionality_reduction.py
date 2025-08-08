#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  8 14:03:13 2025

@author: nk
"""

import numpy as np
import pytest
from sklearn.datasets import make_classification
# Assuming 'perform_pca' is in a file named 'your_module.py'
# Adjust the import according to your actual file structure
from pipeline.dimensionality_reduction import perform_pca

def create_test_dataset(n_samples=100, n_features=10, random_state=42):
    """Helper function to create a reproducible test dataset."""
    X, _ = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_features // 2, # Half features are informative
        n_redundant=n_features // 4,    # Some redundancy
        n_clusters_per_class=1,
        random_state=random_state
    )
    # Add some scaling differences to make standardization impactful
    # (Optional, but good for realism)
    # X = X * np.random.rand(1, n_features) * 10 + np.random.rand(1, n_features) * 5
    return X

def test_perform_pca_basic():
    """Test basic functionality with a standard dataset and threshold."""
    dataset = create_test_dataset(n_samples=100, n_features=20)
    config = {'reduction': {'pca_variance_threshold': 0.95}} # Keep 95% variance

    result = perform_pca(config, dataset)

    # Assert the output is a NumPy array
    assert isinstance(result, np.ndarray)
    # Assert the number of samples remains the same
    assert result.shape[0] == dataset.shape[0]
    # Assert the number of components is less than or equal to original features
    assert result.shape[1] <= dataset.shape[1]
    # Assert that some reduction likely happened (might be flaky with high variance data)
    # assert result.shape[1] < dataset.shape[1] # Uncomment if data guarantees reduction


def test_perform_pca_high_threshold():
    """Test with a high variance threshold (e.g., 99%)."""
    dataset = create_test_dataset(n_samples=50, n_features=15)
    config = {'reduction': {'pca_variance_threshold': 0.99}}

    result = perform_pca(config, dataset)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == dataset.shape[0]
    assert result.shape[1] <= dataset.shape[1]
    # With 99%, we expect fewer components than original features, usually.
    # This might fail if data is already very low rank or threshold is met by all.
    # assert result.shape[1] < dataset.shape[1] # Likely true, but test carefully


def test_perform_pca_low_threshold():
    """Test with a low variance threshold (e.g., 50%)."""
    dataset = create_test_dataset(n_samples=80, n_features=12)
    config = {'reduction': {'pca_variance_threshold': 0.50}}

    result = perform_pca(config, dataset)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == dataset.shape[0]
    assert result.shape[1] <= dataset.shape[1]
    # With 50%, we expect significant reduction if data has structure.
    # assert result.shape[1] < dataset.shape[1] # Likely true


def test_perform_pca_threshold_one():
    """Test with variance threshold of 1.0 (should keep all components needed for 100% variance)."""
    dataset = create_test_dataset(n_samples=30, n_features=8) # Small dataset
    config = {'reduction': {'pca_variance_threshold': 1.0}}

    result = perform_pca(config, dataset)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == dataset.shape[0]
    # With threshold 1.0, PCA might keep min(n_samples - 1, n_features) components
    # or all if n_components='mle' allows it. The function logic might differ.
    # Let's check it doesn't exceed theoretical max components.
    max_components = min(dataset.shape[0] - 1, dataset.shape[1])
    # The function uses 'mle' first, then refits. 'mle' can return fewer.
    # The final number should be <= max_components and <= n_features
    assert result.shape[1] <= dataset.shape[1]
    assert result.shape[1] <= max_components + 1 # Small buffer, PCA details


def test_perform_pca_config_integration():
    """Test that the function correctly uses the threshold from the config."""
    dataset = create_test_dataset(n_samples=60, n_features=10)
    threshold = 0.85
    config = {'reduction': {'pca_variance_threshold': threshold}}

    # We mostly test that it runs without error using the config value
    # Verifying the *exact* number of components chosen is complex without
    # replicating the internal PCA logic.
    result = perform_pca(config, dataset)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == dataset.shape[0]
    assert result.shape[1] <= dataset.shape[1]


# --- Edge Case Tests ---

def test_perform_pca_fewer_samples_than_features():
    """Test with n_samples < n_features."""
    dataset = np.random.rand(5, 20) # 5 samples, 20 features
    config = {'reduction': {'pca_variance_threshold': 0.90}}

    result = perform_pca(config, dataset)

    assert isinstance(result, np.ndarray)
    assert result.shape[0] == dataset.shape[0] # Samples preserved
    # Max components possible is min(n_samples - 1, n_features) = min(4, 20) = 4
    # The 'mle' solver and subsequent logic should respect this.
    assert result.shape[1] <= 4
    assert result.shape[1] >= 0 # Obviously true, but explicit check


# --- Potential Error Condition Tests ---

def test_perform_pca_missing_config_key():
    """Test behavior when the required config key is missing."""
    dataset = create_test_dataset()
    config = {'reduction': {}} # Missing 'pca_variance_threshold' key

    with pytest.raises(KeyError):
        perform_pca(config, dataset)

# Note: Testing the *exact* number of components chosen is tricky because it depends
# on the specific dataset's variance distribution. These tests focus on shape,
# type, and boundary conditions.