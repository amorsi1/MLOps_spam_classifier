# Spam checking using vector embeddings

Filtering spam mail is an arms race between hackers and security teams, but most spam emails in my inbox look really obviously to spot. Using vectors to represent the text content of the emails,
I found that even a really simple classifier could get an F1 score of 0.97 on a clean dataset.

As part of the [MLOps-zoomcamp](https://github.com/DataTalksClub/mlops-zoomcamp) course, I've developed this into an end-to-end cloud pipeline that lets you simply type in some text and classify
whether or not it's spam. 

## Try it out!
[Link](https://amuveqngiu.us-east-1.awsapprunner.com/) 

![MLOPs-spam-clasifier-demo](https://github.com/user-attachments/assets/73fd4bef-4ef6-45f7-8fee-6097120f4670)


## Architecture
<img width="2400" height="1148" alt="MLOps spam_ham architecture" src="https://github.com/user-attachments/assets/a660a559-7d8c-42ee-a1a8-4de40d811a58" />

## Installation instructions
Clone the repo and install dependencies
```
git clone https://github.com/amorsi1/MLOps_spam_classifier
pip install pipenv
cd MLOps_spam_classifier.git
pipenv install 
```
a `.env` file is used to centralize environmental variables, before running any code locally make sure to create this file and populate it with the following variables:
```.env
MLFLOW_TRACKING_URI=http://mlflow-server:8080 
MLFLOW_EXPERIMENT_NAME=spam-classifier
MLFLOW_MODEL_NAME=lr-model
EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2
```
With that set, you can run the training container, mlflow server container, and webapp container using docker compose:
```bash
docker-compose up --build
```

or use the makefile to do the same (the makefile has additional testing capabilities)
```bash
make build
make up
```
## Dataset
A subset of the data in this [Phishing email dataset](https://www.kaggle.com/datasets/naserabdullahalam/phishing-email-dataset) from Kaggle was used for model training. The 2 highest quality
datasets were combined and used.

## Testing
Unit testing and integration tests done using pytest. Note that the integration tests will fail if you don't have a docker daemon running

# Course project Self-Evaluation 
Problem description
0 points: The problem is not described
1 point: The problem is described but shortly or not clearly
**2 points: The problem is well described and it's clear what the problem the project solves**
Cloud
0 points: Cloud is not used, things run only locally
2 points: The project is developed on the cloud OR uses localstack (or similar tool) OR the project is deployed to Kubernetes or similar container management platforms
**4 points: The project is developed on the cloud and IaC tools are used for provisioning the infrastructure**
Experiment tracking and model registry
0 points: No experiment tracking or model registry
2 points: Experiments are tracked or models are registered in the registry
**4 points: Both experiment tracking and model registry are used**
Workflow orchestration
0 points: No workflow orchestration

**2 points: Basic workflow orchestration**

4 points: Fully deployed workflow
Model deployment
0 points: Model is not deployed
2 points: Model is deployed but only locally

**4 points: The model deployment code is containerized and could be deployed to cloud or special tools for model deployment are used**

Model monitoring

**0 points: No model monitoring**

2 points: Basic model monitoring that calculates and reports metrics
4 points: Comprehensive model monitoring that sends alerts or runs a conditional workflow (e.g. retraining, generating debugging dashboard, switching to a different model) if the defined metrics threshold is violated
Reproducibility
0 points: No instructions on how to run the code at all, the data is missing
2 points: Some instructions are there, but they are not complete OR instructions are clear and complete, the code works, but the data is missing

**4 points: Instructions are clear, it's easy to run the code, and it works. The versions for all the dependencies are specified.**

Best practices

 **There are unit tests (1 point)**
 
 **There is an integration test (1 point)**
 
 **Linter and/or code formatter are used (1 point)**
 
 **There's a Makefile (1 point)**
 There are pre-commit hooks (1 point)
 There's a CI/CD pipeline (2 points)






