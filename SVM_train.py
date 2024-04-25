import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from joblib import dump
from DataPreprocessing import process_file

def SVM_train(): 
  
    file_path = 'Dataset/train/trainLarge.txt' 
    result_object = process_file(file_path)

    df = pd.DataFrame(result_object, columns=['Label', 'Text'])
    train_df,_ = train_test_split(df, train_size=0.8, random_state=80)
  
    # Converting into numerical data using TF-IDF technique 
    tfidf_vectorizer = TfidfVectorizer(max_features=5000, stop_words='english') 

    X_train_tfidf = tfidf_vectorizer.fit_transform(train_df['Text'])

    # Extracting labels for training and testing
    y_train = train_df['Label']  
 
    svm_model = SVC(C = 1.0, kernel='rbf', gamma='scale') 
    svm_model.fit(X_train_tfidf, y_train)

    return svm_model, tfidf_vectorizer



def save_model(svm_model,tfidf_vectorizer): 
    folder_path = 'saved_models'
    save_path = f'{folder_path}/svm_saved_model.joblib'
    dump((svm_model, tfidf_vectorizer), save_path) 

model,tfidf_vectorizer =  SVM_train()
save_model(model, tfidf_vectorizer)
