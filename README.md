# Survey Clustering Pipeline 
*WIP - ReadMe incomplete!!!*

As part of a previous job interview, I was tasked to materialize a pipeline for ingesting, processing and performing clustering on a dataset that consisted of answers to questions in a survey. 
I decided to keep the project and in the extra time I had since then, expand upon its structure and deployment.

## The Objective

Given a dataset that corresponds to participants' answers to a series of survey questions, the goal is to perform a segmentation task which can provide insights on the audience of the survey.
The dataset includes answers from two different surveys. The questions of the Core survey, every participant has answered. For the questions of the Recontact survey, only a number of the initial participants have answers.
The pipeline will have to deal with these missing datapoints and process the features so that it can effectively perform the clustering task. 

* A robust, mathematically valid ML Pipeline
* Results must be interpretable and provide useful information
* Configurable and parametric
* Ready for production
* Supported by unit & integration tests

## Architecture

The architecture is modular to ensure scalability and ease of deployment, using a containerized approach with Docker:

<img width="681" height="531" alt="survey_clustering_skl drawio" src="https://github.com/user-attachments/assets/9709d888-8b68-4b10-a42f-aae17b599f82" />

project/ \
├── config/ \
│   └── config.yml # Example config stored \
├── data/ \
│   └── dataset.xlsx  # Example dataset stored \
├── models/ \
│   └── trained_model.keras # Example models stored \
├── models_training/ \
│   ├── train_model_synthesize.py \
│   └── train_model_reduce.py \
├── pipeline/ \
│   ├── data_load.py \
│   ├── process_missing_data.py \
│   ├── feature_selection.py \
│   ├── dimensionality_reduction.py \
│   └── clustering.py \
├── tests/ \
│   └── test_modules.py  # Test modules to cover the pipeline \
├── main.py \
├── app.py \
├── requirements.txt \
└── Dockerfile \

## Codebase:

- "appmain.py" : Main script for launching fastAPI application. 
- "main.py" : Main script for calling the pipeline. 
- "pipeline/" :
   - "data_load.py" : Module for loading the dataset from the specified data source.
   - "process_missing_data.py" : Module to handle the missing data through the specified strategy.
   - "feature_selection.py" : Module to perform feature selection based on statistics & metrics on the dataset.
   - "dimensionality_reduction.py" : Module to perform dimensionality reduction using the specified approach.
   - "clustering.py" : Module to perform the clustering and generate the output files.
- "models_training/" :
   - "train_model_synthesize.py" : Script for building and training a deep learning model for synthesizing missing data.
   - "train_model_reduce.py" : Script for building and training a deep learning model for performing dimensionality reduction.

