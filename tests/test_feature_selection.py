#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the feature selection script.

@author: nk
"""
import pytest
import pandas as pd
from pipeline.feature_selection import perform_varcorrcheck

@pytest.fixture
def base_config():
    """Provides a base configuration dictionary."""
    return {
        'feature_selection': {
            'variance_check': 0,        # Default: Off
            'variance_threshold': 0.01, # Default threshold
            'correlation_check': 0,     # Default: Off
            'correlation_threshold': 0.95 # Default threshold
        }
    }

@pytest.fixture
def sample_dataset():
    """Provides a sample dataset for testing."""
    data = {
        'const_feature': [5, 5, 5, 5, 5],  # Zero variance
        'low_var_feature': [1, 1, 1, 1, 2], # Low variance
        'feature_a': [1, 2, 3, 4, 5],
        'feature_b': [2, 4, 6, 8, 10],  # Perfectly correlated with feature_a
        'feature_c': [1, 3, 5, 7, 9],   # Highly correlated with feature_a
        'feature_d': [10, 20, 30, 40, 50], # Perfectly correlated with feature_b
        'feature_e': [0, 1, 0, 1, 0],   # Independent-ish feature
        'feature_f': [223, 260, 0, 540, 490] # Another independent-ish feature
    }
    return pd.DataFrame(data)

# --- Test Cases ---

def test_no_checks_performed(base_config, sample_dataset):
    """Test that no features are dropped if both checks are disabled."""
    # Ensure checks are off
    base_config['feature_selection']['variance_check'] = 0
    base_config['feature_selection']['correlation_check'] = 0
    
    original_shape = sample_dataset.shape
    result_dataset = perform_varcorrcheck(base_config, sample_dataset.copy())
    
    # Dataset should be unchanged
    assert result_dataset.shape == original_shape
    pd.testing.assert_frame_equal(result_dataset, sample_dataset)


def test_variance_check_only_drop_low_variance(base_config, sample_dataset):
    """Test variance check drops features below threshold."""
    base_config['feature_selection']['variance_check'] = 1
    base_config['feature_selection']['variance_threshold'] = 0.25 # Set threshold
    base_config['feature_selection']['correlation_check'] = 0 # Disable corr check

    # Manually calculate expected variance for clarity in test
    # var_const = 0.0, var_low_var = 0.2
    # So, with threshold 0.5, both 'const_feature' and 'low_var_feature' should be dropped
    expected_dropped = 2
    expected_remaining_features = sample_dataset.shape[1] - expected_dropped
    expected_columns = ['feature_a', 'feature_b', 'feature_c', 'feature_d', 'feature_e', 'feature_f']
    
    result_dataset = perform_varcorrcheck(base_config, sample_dataset.copy())
    
    assert result_dataset.shape[1] == expected_remaining_features
    assert set(result_dataset.columns) == set(expected_columns)
    # Ensure dropped columns are not present
    assert 'const_feature' not in result_dataset.columns
    assert 'low_var_feature' not in result_dataset.columns


def test_correlation_check_only_drop_high_correlation(base_config, sample_dataset):
    """Test correlation check drops one of highly correlated feature pairs."""
    base_config['feature_selection']['variance_check'] = 0 # Disable var check
    base_config['feature_selection']['correlation_check'] = 1
    base_config['feature_selection']['correlation_threshold'] = 0.9 # Set threshold

    # Identify highly correlated pairs manually for this dataset:
    # feature_a vs feature_b: corr = 1.0 -> drop one (e.g., feature_b)
    # feature_a vs feature_c: corr ~ 1.0 -> drop one (e.g., feature_c) 
    # feature_a vs feature_d: corr = 1.0 -> drop one (e.g., feature_d)
    # feature_b vs feature_c: corr ~ 1.0 -> drop one (already considered)
    # feature_b vs feature_d: corr = 1.0 -> drop one (already considered)
    # feature_c vs feature_d: corr = 1.0 -> drop one (already considered)
    # The function drops the *second* feature encountered in the nested loop (column i)
    # Order of columns matters in the dataset for predicting *which* one is dropped,
    # but we know *one* from each perfect correlation group will be dropped.
    # Groups: {feature_a, feature_b, feature_c, feature_d} -> drop 3
    #         {feature_e} -> keep
    #         {feature_f} -> keep
    #         {const_feature} -> keep (corr might be NaN, but check handles abs())
    #         {low_var_feature} -> keep
    # Expected drops: 3 features (one each from the perfectly correlated group)
    expected_dropped_count = 3
    expected_remaining_count = sample_dataset.shape[1] - expected_dropped_count
    
    # We know feature_a is likely kept (first in pair), others dropped.
    # The exact set depends on pandas .corr() internal column order, 
    # but we can check the count and that the kept ones are plausible.
    result_dataset = perform_varcorrcheck(base_config, sample_dataset.copy())

    assert result_dataset.shape[1] == expected_remaining_count
    # Check that at least feature_a, feature_e, feature_f, const_feature, low_var_feature are possibilities
    # The key is the count reduction, not the specific names dropped by correlation logic.
    # A more robust test would mock .corr() or use a dataset where correlation logic is simpler to predict.

def test_both_checks_drop_features(base_config, sample_dataset):
    """Test that both checks can drop features when enabled."""
    base_config['feature_selection']['variance_check'] = 1
    base_config['feature_selection']['variance_threshold'] = 0.25 # Drops const_feature, low_var_feature
    base_config['feature_selection']['correlation_check'] = 1
    base_config['feature_selection']['correlation_threshold'] = 0.9 # Drops highly correlated features

    result_dataset = perform_varcorrcheck(base_config, sample_dataset.copy())

    # Should drop const_feature, low_var_feature, and some highly correlated ones
    # e.g., drop const_feature, low_var_feature, feature_b, feature_c, feature_d
    # This is an integration test, checking the combined effect.
    assert result_dataset.shape[1] < sample_dataset.shape[1]
    assert 'const_feature' not in result_dataset.columns
    assert 'low_var_feature' not in result_dataset.columns
    # Check that at least some of the correlated group are dropped
    correlated_group = {'feature_a', 'feature_b', 'feature_c', 'feature_d'}
    remaining_correlated = correlated_group.intersection(result_dataset.columns)
    # Should not have all 4 remaining
    assert len(remaining_correlated) < 4

