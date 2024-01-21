import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
import tensorflow as tf
import numpy as np
from tensorflow import keras 
from keras.models import Sequential
from keras.layers import Dense 
from keras.optimizers import Adam 
from joblib import dump
from DataPreprocessing import process_file
from keras.models import model_from_json

def backpropagation_train(): 


    train_df = pd.read_csv('Dataset/train_data.csv')

    # Converting into numerical data using TF-IDF technique 
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  

    
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['Text'])


    # Converting sparse matrices to dense arrays
    X_train_dense = X_train_tfidf.toarray()
    

    # Encoding labels 
    label_encoder = LabelEncoder()
    y_train_encoded = label_encoder.fit_transform(train_df['Label'])
 

    
    model = Sequential()
    model.add(Dense(128, input_shape=(X_train_dense.shape[1],), activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))  # single neuron for binary classification

    # Compile the model
    model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    
    model.fit(X_train_dense, y_train_encoded, epochs=30, batch_size=32)

    save_model(model)

    return model 

def save_model(model): 
    model_json = model.to_json()
    with open('saved_models/backpropagation_architecture.json', 'w') as json_file:
        json_file.write(model_json)  
    model.save_weights('saved_models/backpropagation_weights.h5')





