#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 22:19:55 2025

@author: nk
"""
import requests
import pandas as pd
from io import BytesIO
import boto3
from botocore.config import Config

def load_excel_from_config(config: dict) -> pd.DataFrame:
    source = config['data_source']
    source_type = source['type']
    path = source['path']

    if source_type == "url":
        response = requests.get(path)
        response.raise_for_status()
        return pd.read_excel(BytesIO(response.content))

    elif source_type == "local":
        return pd.read_excel(path)

    elif source_type == "s3": # Now pointing to MinIO
    
        # --- CONFIGURE FOR MINIO ---
        # MinIO is used for this Demo project, to showcase S3 compatibility
        # These are from the local MinIO setup
        minio_endpoint = "http://127.0.0.1:9000" # MinIO server address
        access_key = "minioadmin"    # Set in MinIO
        secret_key = "minioadmin"    # Set in MinIO
    
        # Create S3 client configured for MinIO
        s3 = boto3.client(
            's3',
            endpoint_url=minio_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4') # Often needed
        )
        # --- END MINIO CONFIG ---
    
        # Parsing path remains the same if you use s3:// format
        # You might need to adjust if MinIO uses a different convention in your setup
        bucket, key = path.replace("s3://", "").split("/", 1)
    
        # Downloading is the same as S3 API call
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_excel(BytesIO(obj['Body'].read()))
    
    else:
        raise ValueError(f"Unsupported data source type: {source_type}")