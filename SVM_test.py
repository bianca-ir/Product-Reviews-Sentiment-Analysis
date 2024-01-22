from sklearn.metrics import accuracy_score, classification_report
import pandas as pd
from joblib import load

def test_model(model, vectorizer, test_data):
    X_test_tfidf = vectorizer.transform(test_data['Text'])
    y_test = test_data['Label']
    y_pred = model.predict(X_test_tfidf)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")

    # Print more detailed classification report
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred))

# Load the test data
test_data_path = 'Dataset/test_data.csv'
test_df = pd.read_csv(test_data_path, encoding='utf-8')
loaded_model, tfidf_vectorizer = load("saved_models/svm_saved_model.joblib")
# Test the model
test_model(loaded_model, tfidf_vectorizer, test_df)
