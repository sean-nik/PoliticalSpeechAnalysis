#import os
#os.chdir("/Users/sean/Desktop/Syracuse University/Semester 2/IST652 Scripting For Data Analysis/project")

import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))  
from nltk.sentiment.vader import SentimentIntensityAnalyzer

def tokenize_speech(speech):
    transformed_speech = speech.lower() # all lowercase
    # Replace all non alphanumeric characters with spaces
    transformed_speech = re.sub(r'[^@a-zA-Z0-9\s]', ' ', transformed_speech)
    # tokenize
    tokens = nltk.word_tokenize(transformed_speech)
    
    # remove stopwords
    cleaned_tokens = []
    for w in tokens:
        if w not in stopWords:
            cleaned_tokens.append(w)
    return(cleaned_tokens)

# Biden tokenization
# read in speech
biden_speech = ""
with open("biden_dnc20.txt", "r") as file:
    biden_speech = file.read()
    
biden_tokens = tokenize_speech(biden_speech)
biden_tokens[:500]

# Trump tokenization
trump_speech = ""
with open("trump_rnc20.txt", "r") as file:
    trump_speech = file.read()

trump_tokens = tokenize_speech(trump_speech)
trump_tokens[:500]

def plot_word_freqs(list_of_tuples, color, title):
    indices = np.arange(len(list_of_tuples))
    words = [tup[0] for tup in list_of_tuples]
    counts = [tup[1] for tup in list_of_tuples]
    plt.figure(figsize=(8,7))
    plt.barh(indices, counts, color=color)
    plt.yticks(indices, words, rotation='horizontal')
    plt.tight_layout()
    plt.gca().invert_yaxis()
    plt.title(title)
    plt.show()



# Wordclouds
bidenFD = nltk.FreqDist(biden_tokens)
biden_top_words = bidenFD.most_common(30)
plot_word_freqs(biden_top_words, 'b', "Top 30 words in Biden's DNC speech")
wc_biden = WordCloud(background_color="white",width=1000,height=1000, max_words=30,relative_scaling=0,normalize_plurals=False).generate_from_frequencies(dict(biden_top_words))
plt.imshow(wc_biden)

trumpFD = nltk.FreqDist(trump_tokens)
trump_top_words = trumpFD.most_common(30)
plot_word_freqs(trump_top_words, 'r', "Top 30 words in Trump's RNC speech")
wc_trump = WordCloud(background_color="white",width=1000,height=1000, max_words=30,relative_scaling=0,normalize_plurals=False).generate_from_frequencies(dict(trump_top_words))
plt.imshow(wc_trump)


################################
# Sentiment Analysis

def sentiment(speech): 
    sid = SentimentIntensityAnalyzer() 
    sentiment_dict = sid.polarity_scores(speech) 
    print("Overall sentiment dictionary is : ", sentiment_dict) 
    print("sentence was ", sentiment_dict['neg']*100, "% Negative") 
    print("sentence was ", sentiment_dict['neu']*100, "% Neutral") 
    print("sentence was ", sentiment_dict['pos']*100, "% Positive") 
    print("Sentiment overall was", end = " ") 
    if sentiment_dict['compound'] >= 0.01 : 
        print("Positive") 
    elif sentiment_dict['compound'] <= - 0.01 : 
        print("Negative") 
    else : 
        print("Neutral") 

sentiment(biden_speech)
sentiment(trump_speech)


