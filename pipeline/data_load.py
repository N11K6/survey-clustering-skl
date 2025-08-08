#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 22:19:55 2025

@author: nk
"""
import requests
import pandas as pd
from io import BytesIO

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

    elif source_type == "s3":
        # Requires boto3; example: s3://bucket/file.xlsx
        import boto3
        s3 = boto3.client('s3')
        bucket, key = path.replace("s3://", "").split("/", 1)
        obj = s3.get_object(Bucket=bucket, Key=key)
        return pd.read_excel(BytesIO(obj['Body'].read()))

    else:
        raise ValueError(f"Unsupported data source type: {source_type}")