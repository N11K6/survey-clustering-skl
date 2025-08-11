#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Module for processing missing datapoints.

@author: nk
"""

import os
import pandas as pd
import numpy as np
# now we can import normally from sklearn.impute
from sklearn.impute import SimpleImputer

def count_missing(dataset: pd.DataFrame) -> int:
    # Count missing and give an overview of their prevalence in the data
    n_subjects = dataset.shape[0]
    n_features = dataset.shape[1]
    n_total_datapoints = n_subjects*n_features
    nan_count = dataset.isna().sum().sum()
    print(f'{nan_count} NaN datapoints out of {n_total_datapoints} in total.\
          ({np.round(100*nan_count/n_total_datapoints)}%)')
          
    return nan_count

def ignore_missing(dataset: pd.DataFrame) -> pd.DataFrame:
    print('Eliminating all features with missing datapoints.')
    # Ignore missing values
    missing_cols = []
    for col in dataset.columns:
        missing_mask = dataset[col].isna()
        if missing_mask.any():
            missing_cols.append(col)
    dataset_ignored = dataset.drop(missing_cols, axis=1)
    
    return dataset_ignored

def fill_missing(config: dict, dataset: pd.DataFrame) -> pd.DataFrame:
    fill_value = config['missing_data']['fill_value']
    print(f'Replacing missing datapoints with {fill_value}')
    dataset_filled = dataset.fillna(fill_value)
    
    return dataset_filled

def impute_missing(dataset: pd.DataFrame) -> pd.DataFrame:
    print('Imputing most frequent value for missing datapoints.')
    simple_imputer = SimpleImputer(missing_values=np.nan, strategy='most_frequent')
    dataset_imputed = pd.DataFrame(
        simple_imputer.fit_transform(dataset),
        columns=dataset.columns,
        index=dataset.index
        )
    return dataset_imputed

def handle_missing(config, dataset: pd.DataFrame) -> pd.DataFrame:
    '''
    Main process for handling missing datapoints.
    '''
    count_missing(dataset)
    missing_strategy = config['missing_data']['missing_data_strategy']
    print(f'Missing data handled through strategy: {missing_strategy}')
    if missing_strategy == 'ignore':
        dataset = ignore_missing(dataset)
    elif missing_strategy == 'fill':
        dataset = fill_missing(config, dataset)
    elif missing_strategy == 'impute':
        dataset = impute_missing(dataset)
    else:
        raise ValueError('missing_strategy can only be: ignore, fill, impute')
    
    return dataset

if __name__ == "__main__":
    config = {'missing_data':{'missing_data_strategy':'impute'}}
    dataset_path = os.path.join('..', 'data', 'dataset.xlsx')
    dataset = pd.read_excel(dataset_path)
    dataset = handle_missing(config,dataset)



