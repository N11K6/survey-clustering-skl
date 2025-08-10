#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the missing data script.

@author: nk
"""
import pandas as pd
import numpy as np
import pytest
from pipeline.process_missing_data import count_missing, ignore_missing, fill_missing, impute_missing, handle_missing

# --- Fixtures for reusable test data (Integer-specific) ---

@pytest.fixture
def sample_dataset_no_missing_int():
    return pd.DataFrame({
        'A': [1, 2, 3],
        'B': [4, 5, 6],
        'C': [7, 8, 9]
    })

@pytest.fixture
def sample_dataset_with_missing_int():
    return pd.DataFrame({
        'A': [1, np.nan, 3],
        'B': [4, 5, np.nan],
        'C': [np.nan, 8, 9],
        'D': [10, 11, 12] # Column without missing values
    })

# Config Fixtures (Strategies remain the same, just adjust fill_value if needed)
@pytest.fixture
def config_fill_zero():
    return {'missing_data': {'fill_value': 0, 'missing_data_strategy': 'fill'}}

@pytest.fixture
def config_fill_negative():
    return {'missing_data': {'fill_value': -99, 'missing_data_strategy': 'fill'}}

@pytest.fixture
def config_strategy_ignore():
    return {'missing_data': {'missing_data_strategy': 'ignore'}}

@pytest.fixture
def config_strategy_fill():
    return {'missing_data': {'fill_value': -1, 'missing_data_strategy': 'fill'}}

@pytest.fixture
def config_strategy_impute():
    return {'missing_data': {'missing_data_strategy': 'impute'}}

# --- Tests for count_missing ---

def test_count_missing_no_missing_int(capfd, sample_dataset_no_missing_int):
    result = count_missing(sample_dataset_no_missing_int)
    captured = capfd.readouterr()
    assert result == 0
    assert "0 NaN datapoints" in captured.out
    # Check for percentage, allowing for minor formatting variations
    assert "(0.0%)" in captured.out or "(0%)" in captured.out

def test_count_missing_with_missing_int(capfd, sample_dataset_with_missing_int):
    # Dataset: 3 rows, 4 cols = 12 total points. 3 missing (all np.nan)
    result = count_missing(sample_dataset_with_missing_int)
    captured = capfd.readouterr()
    assert result == 3
    assert "3 NaN datapoints" in captured.out
    # 3/12 = 0.25 -> 25%
    assert "(25.0%)" in captured.out or "(25%)" in captured.out

# --- Tests for ignore_missing ---

def test_ignore_missing_no_missing_int(sample_dataset_no_missing_int):
    original_columns = list(sample_dataset_no_missing_int.columns)
    result_df = ignore_missing(sample_dataset_no_missing_int.copy())
    # Should return the same dataframe if no columns have missing values
    pd.testing.assert_frame_equal(result_df, sample_dataset_no_missing_int)
    assert list(result_df.columns) == original_columns

def test_ignore_missing_all_missing_int(sample_dataset_with_missing_int):
    # Make all columns have missing values for this test
    df_all_missing = sample_dataset_with_missing_int.copy()
    df_all_missing.loc[2, 'D'] = np.nan # Now D also has a NaN
    result_df = ignore_missing(df_all_missing.copy())
    # Should return an empty DataFrame as all columns are dropped
    assert result_df.empty
    assert list(result_df.columns) == [] # Columns should be empty list

def test_ignore_missing_partial_missing_int(sample_dataset_with_missing_int):
    # The default fixture already has partial missing (D is clean)
    result_df = ignore_missing(sample_dataset_with_missing_int.copy())
    # Only column D should remain
    expected_df = sample_dataset_with_missing_int[['D']].copy()
    # Reset index if needed, but usually not necessary for drop
    pd.testing.assert_frame_equal(result_df, expected_df)
    assert list(result_df.columns) == ['D']

# --- Tests for fill_missing ---

def test_fill_missing_int_zero(sample_dataset_with_missing_int, config_fill_zero):
    result_df = fill_missing(config_fill_zero, sample_dataset_with_missing_int.copy())
    expected_df = sample_dataset_with_missing_int.fillna(0)
    # fillna might promote int to float, so check_dtype=False
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)
    # Ensure no NaNs remain
    assert not result_df.isna().any().any()

def test_fill_missing_int_negative(sample_dataset_with_missing_int, config_fill_negative):
    result_df = fill_missing(config_fill_negative, sample_dataset_with_missing_int.copy())
    expected_df = sample_dataset_with_missing_int.fillna(-99)
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)
    assert not result_df.isna().any().any()

# --- Tests for impute_missing ---

def test_impute_missing_int_most_frequent():
    # Create a dataset suitable for most_frequent imputation with integers
    df = pd.DataFrame({
        'A': [1, np.nan, 1, 2], # Most frequent in A is 1
        'B': [10, 20, np.nan, 10], # Most frequent in B is 10
        'C': [100, np.nan, np.nan, 100] # Most frequent in C is 100
    })
    result_df = impute_missing(df.copy())

    # SimpleImputer with 'most_frequent' on integers typically returns float
    # Expected result after imputation
    expected_df = pd.DataFrame({
        'A': [1.0, 1.0, 1.0, 2.0], # NaN filled with 1.0 (most frequent)
        'B': [10.0, 20.0, 10.0, 10.0], # NaN filled with 10.0 (most frequent)
        'C': [100.0, 100.0, 100.0, 100.0] # NaN filled with 100.0 (most frequent)
    })
    # Dtype checking is relaxed because SimpleImputer changes int to float
    pd.testing.assert_frame_equal(result_df, expected_df)
    assert not result_df.isna().any().any()

# --- Tests for handle_missing ---

def test_handle_missing_ignore_int(sample_dataset_with_missing_int, config_strategy_ignore):
    result_df = handle_missing(config_strategy_ignore, sample_dataset_with_missing_int.copy())
    expected_df = ignore_missing(sample_dataset_with_missing_int.copy())
    pd.testing.assert_frame_equal(result_df, expected_df)

def test_handle_missing_fill_int(sample_dataset_with_missing_int, config_strategy_fill):
    result_df = handle_missing(config_strategy_fill, sample_dataset_with_missing_int.copy())
    expected_df = fill_missing(config_strategy_fill, sample_dataset_with_missing_int.copy())
    # fillna can change int to float
    pd.testing.assert_frame_equal(result_df, expected_df, check_dtype=False)

def test_handle_missing_impute_int(sample_dataset_with_missing_int, config_strategy_impute):
    result_df = handle_missing(config_strategy_impute, sample_dataset_with_missing_int.copy())
    expected_df = impute_missing(sample_dataset_with_missing_int.copy())
    # SimpleImputer changes dtypes, so check_dtype=False
    pd.testing.assert_frame_equal(result_df, expected_df)
    # Ensure no NaNs remain
    assert not result_df.isna().any().any()

def test_handle_missing_invalid_strategy_int(sample_dataset_no_missing_int):
    invalid_config = {'missing_data': {'missing_data_strategy': 'invalid_strategy'}}
    with pytest.raises(ValueError, match=r"missing_strategy can only be: ignore, fill, impute"):
        handle_missing(invalid_config, sample_dataset_no_missing_int.copy())

