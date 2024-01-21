from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import pandas as pd
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
#from DataPreprocessing import process_file


def test_model():
    train_df = pd.read_csv('Dataset/test_data.csv')
    test_df = pd.read_csv('Dataset/test_data.csv')

    model_path = 'saved_models/svm_model.joblib'  # to be replaced with your own local path

    # Load the model
    model = load(model_path)


    # Recreate the TF-IDF vectorizer and fit on the test data
    tfidf_vectorizer = TfidfVectorizer(max_features=5000)
    X_test_tfidf = tfidf_vectorizer.fit_transform(test_df['Text'])

    # Converting sparse matrices to dense arrays if needed
    X_test_dense = X_test_tfidf.toarray()

    # Create a new label encoder and fit on the training labels
    label_encoder = LabelEncoder()
    label_encoder.fit(train_df['Label'])
    # Encoding labels for test data
    y_test_encoded = label_encoder.transform(test_df['Label'])

    # Predictions on test data
    predictions_str = model.predict(X_test_dense)
    predictions_numeric = label_encoder.transform(predictions_str)  # Convert to numeric


    # Evaluate the model
    accuracy = accuracy_score(y_test_encoded, predictions_numeric)
    precision = precision_score(y_test_encoded, predictions_numeric)
    recall = recall_score(y_test_encoded, predictions_numeric)
    f1 = f1_score(y_test_encoded, predictions_numeric)
    confusion = confusion_matrix(y_test_encoded, predictions_numeric)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion)




#file_path = 'D:\school-projects\year4sem1\Product-Reviews-Sentiment-Analysis/trainLarge.txt'  # to be replaced with your own local path
#result_object = process_file(file_path)
#df = pd.DataFrame(result_object, columns=['Label', 'Text'])
#train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

#save the training and testing data
#train_df.to_csv('train_data.csv', index=False)
#test_df.to_csv('test_data.csv', index=False)


# Test the model
print("testing")
test_model()

