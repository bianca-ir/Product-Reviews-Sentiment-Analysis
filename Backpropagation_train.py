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

    model.fit(X_train_dense, y_train_encoded, epochs=30, batch_size=32, validation_split=0.1)

    return model, tfidf_vectorizer, label_encoder

def save_model(model, tfidf_vectorizer, label_encoder, model_path='saved_models/backpropagation_best_model', vectorizer_path='saved_models/tfidf_bp_vectorizer.joblib', encoder_path='saved_models/label_encoder.joblib'):
    model.save(f'{model_path}.h5')

    dump(tfidf_vectorizer, vectorizer_path)
    dump(label_encoder, encoder_path)

model, tfidf_vectorizer, label_encoder = backpropagation_train()
save_model(model, tfidf_vectorizer, label_encoder)



