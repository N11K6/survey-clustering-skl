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
from tensorflow.keras.models import load_model

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

def synthesize_missing(config: dict, dataset: pd.DataFrame) -> pd.DataFrame:
    '''
    Use trained autoencoder to fill a survey dataset with synthesized data.
    '''
    def prepare_data(dataset: pd.DataFrame):
        '''
        Split the dataset to:
            X -> Core questions 
            y -> Recontact questions
            train -> Entries where Recontact is known
            missing -> Entries without Recontact, to be synthesized
        '''
        dataset_train = dataset.dropna()
        dataset_missing = dataset[dataset.isna().any(axis=1)]
        re_cols_lst = [col for col in dataset.columns if 'core_re' in col]
        dataset_X = dataset_train.drop(re_cols_lst, axis=1).reset_index(drop=True)
        dataset_y = dataset_train[re_cols_lst].reset_index(drop=True)
        dataset_X_missing = dataset_missing.drop(re_cols_lst, axis=1).reset_index(drop=True)
        
        return dataset_X, dataset_y, dataset_X_missing
    
    model_name = config['missing_data']['model_name']
    # Break down dataset to only synthesize on the missing data
    dataset_X, dataset_y, dataset_X_missing = prepare_data(dataset)
    # Load the saved autoencoder
    model_path = os.path.join('models', f'{model_name}.keras')
    loaded_model = load_model(model_path)
    # Synthesize the missing data
    print('Using trained model for data synthesis.')
    results = loaded_model.predict(dataset_X_missing)
    # Create Pandas DF and round to [0 1] format
    dataset_y_missing = pd.DataFrame(data = results, columns = dataset_y.columns).round()
    # Arrange the data for concatenation
    dataset_known = pd.concat([dataset_X, dataset_y], axis=1)
    dataset_synthesized = pd.concat([dataset_X_missing, dataset_y_missing], axis=1)
    # Concatenate synthesized with the known data to provide full dataset
    dataset_full = pd.concat([dataset_known,dataset_synthesized]).reset_index(drop=True)
    
    return dataset_full


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
    elif missing_strategy == 'synthesize':
        dataset = synthesize_missing(config, dataset)
    else:
        raise ValueError('missing_strategy can only be: ignore, fill, impute, synthesize')
    
    return dataset

if __name__ == "__main__":
    config = {'missing_data':{'missing_data_strategy':'synthesize',
                              'model_name':'model_synthesize'}}
    dataset_path = os.path.join('..', 'data', 'dataset.xlsx')
    dataset = pd.read_excel(dataset_path)
    dataset = handle_missing(config,dataset)



