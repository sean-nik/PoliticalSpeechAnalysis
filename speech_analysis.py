import os
os.chdir("/Users/sean/Desktop/Syracuse University/Semester 2/IST652 Scripting For Data Analysis/project")

import nltk
from nltk.tokenize import word_tokenize
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import numpy as np
import re
import matplotlib.pyplot as plt
from wordcloud import WordCloud
from nltk.corpus import stopwords
stopWords = set(stopwords.words('english'))  
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
from tweettokenization import visualize_bigram
from tweettokenization import plot_word_freqs

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


# generate frequency distribution
bidenFD = nltk.FreqDist(biden_tokens)
biden_top_words = bidenFD.most_common(30)
# plot top words as bar chart
plot_word_freqs(biden_top_words, 'b', "Top 30 words in Biden's DNC speech")
# plot top words as wordcloud
wc_biden = WordCloud(background_color="white",width=1000,height=1000, max_words=30,relative_scaling=0,normalize_plurals=False).generate_from_frequencies(dict(biden_top_words))
plt.imshow(wc_biden)

trumpFD = nltk.FreqDist(trump_tokens)
trump_top_words = trumpFD.most_common(30)
# plot top words as bar chart
plot_word_freqs(trump_top_words, 'r', "Top 30 words in Trump's RNC speech")
# plot top words as wordcloud
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



################################
# Bigrams
dnc_finder = BigramCollocationFinder.from_words(biden_tokens)
dnc_finder.nbest(BigramAssocMeasures.chi_sq, 30) # top 30 DNC bigrams
dnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30] # bigrams with scores
# plot barchart
plot_word_freqs(dnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], 'b', "Top 30 bigrams in Biden's DNC Speech", "Frequency Score")
# plot network
visualize_bigram(dnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], .6) # democrat network


rnc_finder = BigramCollocationFinder.from_words(trump_tokens)
rnc_finder.nbest(BigramAssocMeasures.raw_freq, 30) # top 30 RNC bigrams
rnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30] # bigrams with scores
# plot barchart
plot_word_freqs(rnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], 'r', "Top 30 bigrams in Trump's RNC Speech", "Frequency Score")
# plot network
visualize_bigram(rnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], 0.6) # republican network
