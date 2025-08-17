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

The architecture is modular to ensure scalability and ease of deployment, using a containerized approach with Docker. 
The containerized app encompasses the following:

- The app communicates via fastAPI's swagger UI. 
- The user will have to upload a "config.yml" file, containing the parameters for the pipeline, including the data source and filename of the dataset to be used. 
- Using the config.yml as input, the pipeline runs and ingests, processes, and performs clustering on the data.
   - If the parameters dictate the usage of a ML model to perform data processing, the model is loaded from the models directory.
- The outputs are the clustered dataset in Excel format, an image containing the visualization of the clusters, and a JSON file containing evaluation metrics for the clustering.

Modules for training the ML models, and unit testing of the pipeline are included in separate directories which aren't containerized ("models_training/" and "tests/").
Likewise the directories "config/" and "data/" include the example config.yml file and dataset.xlsx to use with the pipeline, but are not part of the deployment.

### Deployment Visualization

<img width="681" height="531" alt="survey_clustering_skl drawio" src="https://github.com/user-attachments/assets/9709d888-8b68-4b10-a42f-aae17b599f82" />

### Project Structure

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
└── Dockerfile

## Parameters
Parameters for the pipeline are defined in the "config.yml", which is the only input file the user has to submit to the app.
#### Data Source
Data source can be:
* "local" : Found locally in the
* "url" : Read from a url.
* "s3" : Loaded from S3 storage. In this demo project, MinIO is used to simulate an AWS like storage as a POC.
#### Missing Data Strategy
One of the processing tasks of the pipeline is to handle the missing data exisitng in the survey dataset. This can be:
* "ignore" : Will drop all the features (questions) where there are missing datapoints.
* "fill" : Will replace all the missing datapoints with a user defined fill value.
* "impute" : Will impute the missing datapoints with the most frequent answer to the specific question.
* "synthesize" : Will synthesize the missing data using an ML model that has been trained on complete survey data.
#### Feature Selection
The feature selection processes available are:
* "variance check" : Drops features that display variance below a user specified threshold.
* "correlation check" : For groups of features displaying correlation higher than a user specified threshold, will keep only one feature.
#### Dimensionality Reduction
Performs dimensionality reduction using one of the following approaches:
* "pca" : The classic Primary Component Analysis procedure, keeping only the components that make up for a user specified cummulative variance.
* "encoder" : Uses a trained Encoder ML model to encode the dataset features into a lower dimension, more suitable for the clustering task.
#### Clustering
The clustering algorithm used is HDBSCAN, which does not require a specified number of clusters.
"minimum cluster size" is the only adjustable parameter for this process, specifying the minimum number of entries that can be considered to make up a distinct cluster. 
The user may specify a number according to the size of the dataset analyzed.

## Codebase
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

