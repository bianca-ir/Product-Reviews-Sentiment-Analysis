# Product-Reviews-Sentiment-Analysis

## Overview
This project involves training and testing machine learning models for sentiment analysis. It includes scripts for data preprocessing, training SVM and backpropagation neural network models and testing these models.

##Prerequisites
Before running the experiments, ensure that you have the following installed:

Python 3.x 

Required Python libraries: pandas, scikit-learn, TensorFlow, Keras, joblib, and spacy.
You can install these libraries using the command:`pip install pandas scikit-learn tensorflow keras joblib spacy`

##Dataset
Place your dataset in the Dataset/ directory. The dataset should be in CSV format with columns 'Text' for reviews and 'Label' for sentiments.
This is the dataset we will use for the Data Mining project. The .tsv file contains nearly 3000 Amazon customer reviews (input text), star ratings, date of review, variant and feedback of some Amazon products. 
Sentiment analysis task will be performed on this dataset. 

Source of data: https://www.kaggle.com/datasets/sid321axn/amazon-alexa-reviews/

#Running the Experiments
##Data Preprocessing
First, preprocess your data: `python .\DataPreprocessing.py`

###Training the Models
To train the SVM model: `python .\SVM_train.py`

To train the Backpropagation Neural Network model: `python .\Backpropagation_train.py`

###Testing the Models
After training, you can test the models:

For SVM: `python .\SVM_test.py`

For Backpropagation Neural Network:`python .\Backpropagation_test.py`

##Results

We also added a predictor script, which can be run using the command: `python .\predictor.py --text {your_string_review}`

The results including accuracy, precision, recall, and F1-score will be displayed in the console after running the test scripts.

Results for SVM model:
Accuracy: 0.96
Precision: 0.95
Recall: 0.97
F1 Score: 0.96

Results for Backpropagation: 
Accuracy: 0.81
Precision: 0.82
Recall: 0.78
F1 Score: 0.80
Confusion Matrix:
[[420  85]
 [105 390]]

