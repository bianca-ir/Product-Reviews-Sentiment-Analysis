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

def backpropagation_train(df): 

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Converting into numerical data using TF-IDF technique 
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)  

    
    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['Text'])

    
    X_test_tfidf = tfidf_vectorizer.transform(test_df['Text'])

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

    # Train the model 
   # history = model.fit(
    #    X_train_dense, y_train_encoded,
    #    epochs=5,
    #    batch_size=32,
    #    validation_data=(X_test_dense, y_test_encoded)
   # )
    
    model.fit(X_train_tfidf, y_train_encoded, epochs=5, batch_size=32)

    return model 

    


