#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Main script for running the pipeline

@author: nk
"""
import os
import yaml
from pipeline import data_load, process_missing_data, feature_selection, dimensionality_reduction, clustering

def run_pipeline(config: dict):
    dataset = data_load.load_excel_from_config(config)
    dataset_full = process_missing_data.handle_missing(config,dataset)
    dataset_selected = feature_selection.perform_varcorrcheck(config,dataset_full)
    data_reduced = dimensionality_reduction.reduce(config, dataset_selected)
    data_clustered, metrics_dict, cluster_plot = clustering.perform_clustering(config, data_reduced)
    dataset_clustered = dataset_selected.copy()
    dataset_clustered['cluster'] = data_clustered
    return dataset_clustered, metrics_dict, cluster_plot

if __name__ == "__main__":
    config_path = os.path.join('config', 'config.yml')
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    dataset_clustered, metrics_dict, cluster_plot = run_pipeline(config)
    cluster_plot.show()
    