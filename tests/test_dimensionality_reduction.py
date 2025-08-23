#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the dimensionalit reduction script.

@author: nk
"""
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from pipeline.dimensionality_reduction import perform_pca, perform_encoding, reduce

# Sample config and dataset for testing
@pytest.fixture
def sample_config():
    return {
        'reduction': {
            'approach': 'pca',
            'pca_variance_threshold': 0.95,
            'model_name': 'test_autoencoder'
        },
        'feature_selection': {
            'variance_check': 0,
            'correlation_check': 0
        }
    }

@pytest.fixture
def sample_dataset():
    # 10 samples, 5 features
    return np.random.rand(10, 5)

@pytest.fixture
def pca_config():
    return {
        'reduction': {
            'approach': 'pca',
            'pca_variance_threshold': 0.95
        },
        'feature_selection': {}
    }

@pytest.fixture
def encoder_config():
    return {
        'data_source': {
            'type': 'local',
            },
        'reduction': {
            'approach': 'encoder',
            'model_name': 'test_encoder'
        },
        'feature_selection': {
            'variance_check': 0,
            'correlation_check': 0
        }
    }

# ========================
# Test: perform_pca
# ========================
def test_perform_pca(pca_config, sample_dataset):
    with patch('pipeline.dimensionality_reduction.PCA') as mock_pca:
        # Simulate PCA behavior
        mock_pca_instance = MagicMock()
        mock_pca_instance.explained_variance_ratio_ = [0.6, 0.3, 0.05, 0.03, 0.02]  # Cumulative: 0.6, 0.9, 0.95, 0.98...
        mock_pca_instance.fit_transform.return_value = sample_dataset[:, :3]  # Simulate 3 components
        mock_pca.return_value = mock_pca_instance

        result = perform_pca(pca_config, sample_dataset)

        # Check that PCA was called with no initial n_components
        mock_pca.assert_any_call()  # First PCA() call
        assert mock_pca.call_count == 2  # One for full, one for optimal
        assert result.shape[1] <= sample_dataset.shape[1]
        assert result.shape[0] == sample_dataset.shape[0]

def test_perform_pca_optimal_components(pca_config, sample_dataset):
    # Real PCA run (no mocking) to test logic
    result = perform_pca(pca_config, sample_dataset)

    # Check explained variance accumulation
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(sample_dataset)
    pca_full = PCA()
    pca_full.fit(X_scaled)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)
    desired_var = pca_config['reduction']['pca_variance_threshold']
    expected_n_components = np.count_nonzero(cumvar <= desired_var)  # <= gives indices below threshold

    assert result.shape[1] == expected_n_components

# ========================
# Test: perform_encoding
# ========================
@patch('os.path.join', return_value='models/test_encoder.keras')
@patch('pipeline.dimensionality_reduction.load_model')
def test_perform_encoding(mock_load_model, mock_join, encoder_config, sample_dataset):
    # Mock encoder model
    mock_encoder = MagicMock()
    mock_encoder.predict.return_value = sample_dataset[:, :3]  # Encoded to 3 features
    mock_load_model.return_value = mock_encoder

    with patch('sklearn.preprocessing.StandardScaler.fit_transform') as mock_scaler:
        mock_scaler.return_value = np.random.rand(*sample_dataset[:, :3].shape)
        result = perform_encoding(encoder_config, sample_dataset)

        mock_load_model.assert_called_once_with('models/test_encoder.keras')
        mock_encoder.predict.assert_called_once_with(sample_dataset)
        assert result.shape[0] == sample_dataset.shape[0]
        assert result.shape[1] == 3  # Output matches encoder output

@patch('warnings.warn')
def test_perform_encoding_with_variance_check_warning(mock_warn, encoder_config):
    # Activate variance check
    encoder_config['feature_selection']['variance_check'] = 1
    sample_dataset = np.random.rand(10, 5)

    with patch('pipeline.dimensionality_reduction.load_model') as mock_load_model:
        mock_encoder = MagicMock()
        mock_encoder.predict.return_value = sample_dataset[:, :3]
        mock_load_model.return_value = mock_encoder

        with patch('pipeline.dimensionality_reduction.StandardScaler.fit_transform'):
            perform_encoding(encoder_config, sample_dataset)
            assert mock_warn.called
            assert "incompatible input dimensions" in str(mock_warn.call_args)

# ========================
# Test: reduce
# ========================
def test_reduce_with_pca(pca_config, sample_dataset):
    with patch('pipeline.dimensionality_reduction.perform_pca') as mock_pca:
        mock_pca.return_value = sample_dataset[:, :3]
        result = reduce(pca_config, sample_dataset)
        mock_pca.assert_called_once_with(pca_config, sample_dataset)
        assert result.shape[1] == 3

def test_reduce_with_encoder(encoder_config, sample_dataset):
    with patch('pipeline.dimensionality_reduction.perform_encoding') as mock_encoding:
        mock_encoding.return_value = sample_dataset[:, :2]
        result = reduce(encoder_config, sample_dataset)
        mock_encoding.assert_called_once_with(encoder_config, sample_dataset)
        assert result.shape[1] == 2

def test_handle_missing_invalid_approach(sample_dataset):
    invalid_config = {'reduction': {'approach': 'invalid_approach'}}
    with pytest.raises(ValueError, match=r"Dimensionality reduction approach can only be: pca, encoder"):
        reduce(invalid_config, sample_dataset.copy())
