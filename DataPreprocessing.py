import spacy
from spacy.lang.en.stop_words import STOP_WORDS
import pandas as pd

def process_file(file_path):
    
    nlp = spacy.load("en_core_web_sm")
    with open(file_path, 'r', encoding='utf-8') as file:
        reviews =[]
        for line in file:
            parts = line.split(maxsplit=1)
            if len(parts) == 2:
                label, review = parts
                review = review.strip().lower()  # Lowercase the text and remove leading/trailing whitespaces
                cleaned_text = ' '.join([token.text for token in nlp(review) if token.text.lower() not in STOP_WORDS]) #remove stop words
                reviews.append((label, cleaned_text))
    return reviews

#file_path = 'C:/Users/andre/Documents/GitHub/Product-Reviews-Sentiment-Analysis/Dataset/train/trainMedium.txt' #to be replaced with your own local path
file_path = 'Dataset/train/trainLarge.txt'
result_object = process_file(file_path)
#print(result_object[0]) #print(result_object[0][1]['cats']['pos'])

# Convert data to a Pandas DataFrame
df = pd.DataFrame(result_object, columns=['Label', 'Text'])

#this is just to see the data we are going to feed to the model
#output_file_path = 'C:/Users/andre/Documents/GitHub/Product-Reviews-Sentiment-Analysis/Dataset/train/output_file.txt'
#df.to_csv(output_file_path, sep='\t', index=False)



