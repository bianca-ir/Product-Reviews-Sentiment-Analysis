import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from joblib import dump

def SVM_train(df): 
    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

    # Converting into numerical data using TF-IDF technique 
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) 

    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['Text'])

    # Extracting labels for training and testing
    y_train = train_df['Label']
    
 
    svm_model = SVC(C = 1.0, kernel='rbf', gamma='scale') # better results with rbf kernel 
    svm_model.fit(X_train_tfidf, y_train)

    return svm_model 

    

