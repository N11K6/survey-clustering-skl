#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script for running the pipeline with fastAPI

@author: nk
"""
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse
import yaml
import io
import json
import zipfile
from pipeline.data_load import load_excel_from_config
from main import run_pipeline

app = FastAPI(title="Clustering API")

@app.post("/cluster/")
async def run_clustering(config_file: UploadFile = File(...)):
    try:
        # 1. Parse YAML config
        config_content =  await config_file.read()
        config = yaml.safe_load(config_content)

        # 2. Load Excel data based on config
        dataset = load_excel_from_config(config)

        # 4. Run clustering
        dataset_clustered, metrics, fig = run_pipeline(config, dataset)

        # 5. In-memory files
        excel_buffer = io.BytesIO()
        dataset_clustered.to_excel(excel_buffer, index=False)
        excel_buffer.seek(0)
        
        image_buffer = io.BytesIO()
        fig.savefig(image_buffer, format='png', bbox_inches='tight')
        image_buffer.seek(0)
        
        json_buffer = io.BytesIO()
        json_buffer.write(json.dumps(metrics, indent=2).encode())
        json_buffer.seek(0)

        # 6. Create ZIP
        zip_buffer = io.BytesIO()
        with zipfile.ZipFile(zip_buffer, 'w') as zf:
            zf.writestr("clustered_data.xlsx", excel_buffer.read())
            zf.writestr("cluster_plot.png", image_buffer.read())
            zf.writestr("metrics.json", json_buffer.read())
        zip_buffer.seek(0)

        # 7. Return results
        return StreamingResponse(
            zip_buffer,
            media_type="application/x-zip-compressed",
            headers={"Content-Disposition": "attachment; filename=results.zip"}
            )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))