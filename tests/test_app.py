#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Tests for the app.py script (Running the pipeline via fastAPI)

@author: nk
"""
from fastapi.testclient import TestClient
from unittest.mock import patch
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import io
import zipfile
import json
matplotlib.use('Agg') # Use the Agg backend for testing to avoid GUI issues
from app import app  # Import FastAPI app

client = TestClient(app)

def test_cluster_endpoint_success():
    # 1. Create mock data and plot
    mock_dataset = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4]}) # Example DataFrame
    mock_clustered_dataset = pd.DataFrame({'col1': [1, 2], 'col2': [3, 4], 'Cluster': [0, 1]}) # Example clustered DataFrame
    mock_metrics = {"silhouette_score": 0.5} # Example metrics

    # Create a simple mock plot
    fig, ax = plt.subplots()
    ax.plot([1, 2], [3, 4])
    # Ensure the figure is closed after use in tests to prevent resource warnings
    # We'll pass the figure object itself, but be mindful of cleanup if needed later.

    # 2. Define the mock config content (structure should match what your real config looks like)
    # Even though we mock the functions, the config might still be partially parsed or validated.
    mock_config_content = """
    data_file: "test_data.xlsx"
    sheet_name: "Sheet1"
    features: ["col1", "col2"]
    # Add other keys your functions might check
    """

    # 3. Use patch to replace the real functions with mocks
    with patch('app.load_excel_from_config') as mock_load_excel, \
         patch('app.run_pipeline') as mock_run_pipeline:

        # 4. Configure the mocks to return our predefined values
        mock_load_excel.return_value = mock_dataset
        mock_run_pipeline.return_value = (mock_clustered_dataset, mock_metrics, fig)

        # 5. Prepare the file upload (using the mock config content)
        files = {
            "config_file": ("mock_config.yaml", mock_config_content, "application/x-yaml")
        }

        # 6. Make the request
        response = client.post("/cluster/", files=files)

        # 7. Assert the response
        assert response.status_code == 200, f"Expected 200, got {response.status_code}. Detail: {response.text}"
        assert response.headers["content-type"] == "application/x-zip-compressed"
        assert "attachment; filename=results.zip" in response.headers["content-disposition"]

        # 8. Optional: Validate the ZIP content
        content = response.content
        with zipfile.ZipFile(io.BytesIO(content)) as z:
            file_list = z.namelist()
            assert "clustered_data.xlsx" in file_list, f"Missing clustered_data.xlsx in {file_list}"
            assert "cluster_plot.png" in file_list, f"Missing cluster_plot.png in {file_list}"
            assert "metrics.json" in file_list, f"Missing metrics.json in {file_list}"

            # Read and validate metrics.json content
            with z.open("metrics.json") as metrics_file:
                metrics_data = json.load(metrics_file)
                assert metrics_data == mock_metrics, f"Metrics mismatch: {metrics_data} != {mock_metrics}"

        plt.close(fig)
