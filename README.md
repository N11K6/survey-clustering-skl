# Survey Clustering Pipeline

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

## Components

* app.py
* main.py
* /config/
  * config.yml
* /pipeline/
  * data_load.py
  * process_missing_data.py
  * feature_selection.py
  * dimensionality_reduction.py
  * clustering.py
