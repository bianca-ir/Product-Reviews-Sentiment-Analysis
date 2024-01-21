import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from joblib import dump
from DataPreprocessing import process_file

def SVM_train(): 
  
    train_df = pd.read_csv('Dataset/train_data.csv')
  
    # Converting into numerical data using TF-IDF technique 
    tfidf_vectorizer = TfidfVectorizer(max_features=5000) 

    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['Text'])

    # Extracting labels for training and testing
    y_train = train_df['Label']  
 
    svm_model = SVC(C = 10, kernel='linear', gamma='scale') 
    svm_model.fit(X_train_tfidf, y_train)

    return svm_model 

    
def save_model(svm_model): 
    folder_path = 'saved_models'
    save_path = f'{folder_path}/svm_saved_model.joblib'
    dump(svm_model, save_path) 
