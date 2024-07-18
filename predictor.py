import argparse
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--text", type=str, required=True, nargs="*")
args = parser.parse_args()
text = " ".join(args.text)

loaded_model, tfidf_vectorizer = load("saved_models/svm_saved_model.joblib")

# Load training data for fitting the vectorizer 

test_df = pd.read_csv('Dataset/test_data.csv')


X_test_tfidf = tfidf_vectorizer.transform(test_df['Text'])

X_test_tfidf = tfidf_vectorizer.transform([text])
X_test_dense = X_test_tfidf.toarray()
y_pred = loaded_model.predict(X_test_dense)

result = y_pred[0]
if result == "__label__2":
    print("The text was positive :)")
else:
    print("The text was negative :(")
