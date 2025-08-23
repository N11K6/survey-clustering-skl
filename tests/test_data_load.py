#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Aug 10 15:16:37 2025

@author: nk
"""
import pytest
import pandas as pd
from io import BytesIO
from unittest.mock import patch, MagicMock, ANY
import sys
import os

# Add the directory containing script1.py to the Python path
# Adjust the path as necessary depending on your project structure
sys.path.insert(0, os.path.dirname(__file__))

# Import the function to test
from pipeline.data_load import load_excel_from_config

# --- Sample data for mocking ---
SAMPLE_CONFIG_LOCAL = {
    'data_source': {
        'type': 'local',
        'filename': 'data.xlsx'
    }
}

SAMPLE_CONFIG_S3 = {
    'data_source': {
        'type': 's3',
        'filename': 'data.xlsx',
        's3_path': 's3://my-test-bucket/path/to/',
        's3_address': 'http://127.0.0.1:9000',
        's3_id': 'minioadmin',
        's3_key': 'minioadmin'
    }
}

SAMPLE_CONFIG_UNSUPPORTED = {
    'data_source': {
        'type': 'unsupported_type',
        'filename': 'some_file.xlsx',
        'path': 'some_path'
    }
}

# Sample Excel content (as bytes) for mocking responses
# In practice, you might load a small test Excel file or create bytes programmatically
# For simplicity here, we'll mock the BytesIO content directly
SAMPLE_EXCEL_BYTES = b"Mocked Excel file content bytes"
SAMPLE_DF = pd.DataFrame({'col1': [1, 2], 'col2': ['a', 'b']}) # Example DataFrame structure

# --- Tests ---

def test_load_excel_from_local():
    """Test loading Excel data from a local file."""
    with patch('pipeline.data_load.pd.read_excel') as mock_read_excel:
        # Setup mock read_excel to return a sample DataFrame
        mock_read_excel.return_value = SAMPLE_DF

        # Call the function
        result_df = load_excel_from_config(SAMPLE_CONFIG_LOCAL)
        
        # Assertions
        mock_read_excel.assert_called_once_with(os.path.join('data',SAMPLE_CONFIG_LOCAL['data_source']['filename']))
        pd.testing.assert_frame_equal(result_df, SAMPLE_DF)

def test_load_excel_from_s3():
    """Test loading Excel data from S3 (MinIO)."""
    with patch('pipeline.data_load.boto3.client') as mock_boto3_client, \
         patch('pipeline.data_load.pd.read_excel') as mock_read_excel:

        # Setup mock S3 client and get_object response
        mock_s3_client = MagicMock()
        mock_boto3_client.return_value = mock_s3_client
        mock_s3_object = {'Body': BytesIO(SAMPLE_EXCEL_BYTES)}
        mock_s3_client.get_object.return_value = mock_s3_object

        # Setup mock read_excel to return a sample DataFrame
        mock_read_excel.return_value = SAMPLE_DF

        # Call the function
        result_df = load_excel_from_config(SAMPLE_CONFIG_S3)

        # Assertions
        # Check boto3.client was called with correct MinIO config
        mock_boto3_client.assert_called_once_with(
            's3',
            endpoint_url="http://127.0.0.1:9000",
            aws_access_key_id="minioadmin",
            aws_secret_access_key="minioadmin",
            config=ANY # We can check the Config object more specifically if needed
        )
        # Check get_object was called with correct bucket and key
        mock_s3_client.get_object.assert_called_once_with(
            Bucket='my-test-bucket',
            Key='path/to/data.xlsx'
        )
        # Check pd.read_excel was called with BytesIO containing the S3 object body
        mock_read_excel.assert_called_once()
        args, kwargs = mock_read_excel.call_args
        assert isinstance(args[0], BytesIO)
        # The BytesIO content should match what was in the mocked S3 object body
        assert args[0].getvalue() == SAMPLE_EXCEL_BYTES

        pd.testing.assert_frame_equal(result_df, SAMPLE_DF)

def test_load_excel_unsupported_type():
    """Test loading with an unsupported source type raises ValueError."""
    with pytest.raises(ValueError) as exc_info:
        load_excel_from_config(SAMPLE_CONFIG_UNSUPPORTED)

    assert f"Unsupported data source type: {SAMPLE_CONFIG_UNSUPPORTED['data_source']['type']}" in str(exc_info.value)

