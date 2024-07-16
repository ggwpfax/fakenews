# Fake News Detection using Amazon SageMaker
This project focuses on detecting fake news using machine learning models deployed on Amazon SageMaker. The repository contains Jupyter notebooks and Python scripts that guide you through the entire process, from data exploration and preprocessing to model training and evaluation.

# Project Structure
* **Data Exploration and S3 Integration (1) (1).ipynb**: Notebook for exploring the dataset and integrating with Amazon S3 for data storage and retrieval.
* **Model building and evaluation.ipynb:** Notebook for building, training, and evaluating various machine learning models.
* **Text shaping for RNN and Training.ipynb:** Notebook for preprocessing text data to shape it for Recurrent Neural Network (RNN) models and training them.
* **helper.py:** Python script containing helper functions used across various notebooks and scripts.
* **train_keras_lstm.py:** Python script for training a Long Short-Term Memory (LSTM) model using Keras.
* **train_sklearn_nb.py:** Python script for training a Naive Bayes model using scikit-learn.

# Getting Started
## Prerequisites
* Python 3.6 or higher
* Jupyter Notebook
* Amazon Web Services (AWS) account
* Required Python libraries (listed in requirements.txt)

## Installation
1. Clone the repository:
   `git clone https://github.com/yourusername/fake-news-detection-sagemaker.git`
   `cd fake-news-detection-sagemaker`
2. Install the required libraries:
   `pip install -r requirements.txt`
3. Set up your AWS credentials and configure AWS CLI:
   `aws configure`

# Usage
**Data Exploration and S3 Integration:**
* Open Data Exploration and S3 Integration (1) (1).ipynb in Jupyter Notebook.
* Follow the steps to explore the dataset and upload it to an S3 bucket.
  
**Text Shaping and Training:**
* Open Text shaping for RNN and Training.ipynb to preprocess text data and train an RNN model.
* You can also use train_keras_lstm.py to train an LSTM model directly.
  
**Model Building and Evaluation:**
* Open Model building and evaluation.ipynb to build and evaluate various models.
* Alternatively, use train_sklearn_nb.py to train a Naive Bayes model.
