import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from tensorflow.keras.models import load_model
from joblib import load

def load_model_and_preprocessors(model_path, vectorizer_path, encoder_path):
    model = load_model(model_path)

    tfidf_vectorizer = load(vectorizer_path)
    label_encoder = load(encoder_path)

    return model, tfidf_vectorizer, label_encoder

def prepare_test_data(tfidf_vectorizer, label_encoder, file_path):
    test_df = pd.read_csv(file_path)
    X_test = tfidf_vectorizer.transform(test_df['Text']).toarray()
    y_test_encoded = label_encoder.transform(test_df['Label'])
    return X_test, y_test_encoded

def test_model(model, X_test, y_test):
    predictions = model.predict(X_test)
    predictions = (predictions > 0.5).astype(int)

    print_evaluation(y_test, predictions)

def print_evaluation(y_test, predictions):
    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    confusion = confusion_matrix(y_test, predictions)

    print(f"Accuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("Confusion Matrix:")
    print(confusion)

def main():
    model_path = 'saved_models/backpropagation_best_model.h5'
    vectorizer_path = 'saved_models/tfidf_bp_vectorizer.joblib'
    encoder_path = 'saved_models/label_encoder.joblib'
    test_data_path = 'Dataset/test_data.csv'

    model, tfidf_vectorizer, label_encoder = load_model_and_preprocessors(model_path, vectorizer_path, encoder_path)
    X_test, y_test = prepare_test_data(tfidf_vectorizer, label_encoder, test_data_path)
    test_model(model, X_test, y_test)

if __name__ == "__main__":
    main()
