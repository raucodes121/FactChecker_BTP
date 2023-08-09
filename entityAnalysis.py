import spacy
from textblob import TextBlob
import pandas as pd
from tabulate import tabulate

# Load spaCy model for NER
nlp = spacy.load("en_core_web_sm")

# Load NRC-Emotion-Lexicon dataset
emolex_df = pd.read_csv('NRC-Emotion-Lexicon.csv', encoding='utf-8')

# Read input text from file in utf-8 format
with open("speech.txt", "r", encoding="utf-8") as file:
    text = file.read()

text = text.replace("'s", '')
text = text.replace("’s", '')

# Perform named entity recognition (NER)
doc = nlp(text)
entities = {}
for entity in doc.ents:
    if entity.label_ in ["PERSON", "ORG"]:
        entity_text = entity.text.lower()
        if entity_text not in entities:
            entities[entity_text] = []

# Group related sentences for each entity
for sentence in doc.sents:
    for entity in entities.keys():
        if entity in sentence.text.lower():
            entities[entity].append(sentence.text)

# Perform sentiment analysis for each entity
emotions_label = ['Anger', 'Anticipation', 'Disgust', 'Fear', 'Joy', 'Sadness', 'Surprise', 'Trust']

def get_emotions(text):
    emotion_counts = {emotion: 0 for emotion in emotions_label}
    for word in text.split():
        word = word.lower()
        if word in emolex_df['English (en)'].str.lower().values:
            word_emotions = emolex_df[emolex_df['English (en)'].str.lower() == word]
            for emotion in emotions_label:
                emotion_counts[emotion] += word_emotions[emotion].values[0]
    return emotion_counts

entity_emotions_list = []
for entity, sentences in entities.items():
    combined_text = " ".join(sentences)
    blob = TextBlob(combined_text)
    sentiment = blob.sentiment.polarity
    sentiment_label = "Positive" if sentiment > 0 else "Negative" if sentiment < 0 else "Neutral"
    emotion_counts = get_emotions(combined_text)
    entity_emotions = {"Entity": entity, "Sentiment Polarity": sentiment, "Sentiment Label": sentiment_label}
    entity_emotions.update(emotion_counts)
    entity_emotions_list.append(entity_emotions)

# Create pandas DataFrame from list of entity emotions
df = pd.DataFrame(entity_emotions_list)
# Reorder the columns for better alignment
cols = ["Entity", "Sentiment Polarity", "Sentiment Label"] + emotions_label
df = df[cols]

# Define a function for formatting the numbers with the first letter of each emotion
def format_numbers(num):
    return f"{num}{' '*(6-len(str(num)))}"

# Apply the formatting function to each emotion column
for emotion in emotions_label:
    df[emotion] = df[emotion].apply(format_numbers)

# Convert DataFrame to tabulate format and print
tabulate_fmt = 'grid'
print(tabulate(df, headers='keys', tablefmt=tabulate_fmt, numalign=format_numbers))