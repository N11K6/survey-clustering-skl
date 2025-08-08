#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  7 15:40:35 2025

@author: nk
"""

def perform_varcorrcheck(config, dataset):
        
    # Variance check
    if int(config['feature_selection']['variance_check']) == 1:
        variance_threshold = config['feature_selection']['variance_threshold']
        print(f'Performing variance check with threshold {variance_threshold}.')
        num_features_before = dataset.shape[1]
        dataset = dataset.loc[:,dataset.var()>variance_threshold]
        print(f'{num_features_before-dataset.shape[1]} features dropped due to low variance.')
        print(f'{dataset.shape[1]} features remain.')

    # Correlation check
    if int(config['feature_selection']['correlation_check']) == 1:
        correlation_threshold = config['feature_selection']['correlation_threshold']
        print(f'Performing correlation check with threshold {correlation_threshold}.')
        num_features_before = dataset.shape[1]
        corr_matrix = dataset.corr().abs()
        high_corr = set()
        for i in range(len(corr_matrix.columns)):
            for j in range(i):
                if corr_matrix.iloc[i, j] > correlation_threshold:
                    colname = corr_matrix.columns[i]
                    high_corr.add(colname)
        dataset = dataset.drop(columns=high_corr)    
        print(f'{num_features_before-dataset.shape[1]} features dropped due to high correlation.')
        print(f'{dataset.shape[1]} features remain.')

    return dataset
