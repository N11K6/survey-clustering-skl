#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 22:19:55 2025

@author: nk
"""
import os
import pandas as pd
from io import BytesIO
import boto3
from botocore.config import Config

def load_excel_from_config(config: dict) -> pd.DataFrame:
    source = config['data_source']
    source_type = source['type']

    if source_type == "local":
        path = source['local_path']
        return pd.read_excel(path)

    elif source_type == "s3":  # Pointing to MinIO for s3 storage
        # Use environment variables with fallback to config
        path = source['s3_path']
        minio_endpoint = os.getenv('MINIO_ENDPOINT', source.get("s3_address"))
        access_key = os.getenv('MINIO_ACCESS_KEY', source.get("s3_id", "minioadmin"))
        secret_key = os.getenv('MINIO_SECRET_KEY', source.get("s3_key", "minioadmin"))
        
        # Create S3 client configured for MinIO
        s3 = boto3.client(
            's3',
            endpoint_url=minio_endpoint,
            aws_access_key_id=access_key,
            aws_secret_access_key=secret_key,
            config=Config(signature_version='s3v4')
        )
        
        # Parsing path
        bucket, key = path.replace("s3://", "").split("/", 1)
    
        # Downloading is the same as S3 API call
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_excel(BytesIO(obj['Body'].read()))
    
    else:
        raise ValueError(f"Unsupported data source type: {source_type}")