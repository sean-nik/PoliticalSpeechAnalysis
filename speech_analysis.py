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

# Tokenization

# Biden 
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

# Trump 2016 tokenization
trump16_speech = ""
with open("trump_rnc16.txt", "r") as file:
    trump16_speech = file.read()

trump16_tokens = tokenize_speech(trump16_speech)
trump16_tokens[:500]

# Clinton 2016 tokenization
clinton16_speech = ""
with open("clinton_dnc16.txt", "r") as file:
    clinton16_speech = file.read()

clinton16_tokens = tokenize_speech(clinton16_speech)
clinton16_tokens[:500]

# Pence 2020 tokenization
pence20_speech = ""
with open("pence_rnc20.txt", "r") as file:
    pence20_speech = file.read()

pence20_tokens = tokenize_speech(pence20_speech)
pence20_tokens[:500]

# Harris 2020 tokenization
harris20_speech = ""
with open("harris_dnc20.txt", "r") as file:
    harris20_speech = file.read()

harris20_tokens = tokenize_speech(harris20_speech)
harris20_tokens[:500]

################################################################################################
# Frequency Distributions

# generate frequency distribution
bidenFD = nltk.FreqDist(biden_tokens)
biden_top_words = bidenFD.most_common(30)
# plot top words as bar chart
plot_word_freqs(biden_top_words, 'b', "Top 30 words in Biden's 2020 DNC speech")

trumpFD = nltk.FreqDist(trump_tokens)
trump_top_words = trumpFD.most_common(30)
# plot top words as bar chart
plot_word_freqs(trump_top_words, 'r', "Top 30 words in Trump's 2020 RNC speech")

# clinton 2016
clinton16FD = nltk.FreqDist(clinton16_tokens)
clinton16_top_words = clinton16FD.most_common(30)
# plot top words as bar chart
plot_word_freqs(clinton16_top_words, 'b', "Top 30 words in Clinton's 2016 DNC speech")

# trump 2016
trump16FD = nltk.FreqDist(trump16_tokens)
trump16_top_words = trump16FD.most_common(30)
# plot top words as bar chart
plot_word_freqs(trump16_top_words, 'r', "Top 30 words in Trump's 2016 RNC speech")

# harris 2020
harrisFD = nltk.FreqDist(harris20_tokens)
harris_top_words = harrisFD.most_common(30)
# plot top words as bar chart
plot_word_freqs(harris_top_words, 'b', "Top 30 words in Harris' 2020 DNC speech")

# pence 2020
penceFD = nltk.FreqDist(pence20_tokens)
pence20_top_words = penceFD.most_common(30)
# plot top words as bar chart
plot_word_freqs(pence20_top_words, 'r', "Top 30 words in Pence's 2020 RNC speech")

################################################################################################
# Sentiment Analysis

def sentiment(speech): 
    sid = SentimentIntensityAnalyzer() 
    sentiment_dict = sid.polarity_scores(speech) 
    print("Overall sentiment dictionary is : ", sentiment_dict) 
    print("speech was ", sentiment_dict['neg']*100, "% Negative") 
    print("speech was ", sentiment_dict['neu']*100, "% Neutral") 
    print("speech was ", sentiment_dict['pos']*100, "% Positive") 
    print("Sentiment overall was", end = " ") 
    if sentiment_dict['compound'] >= 0.01 : 
        print("Positive") 
    elif sentiment_dict['compound'] <= - 0.01 : 
        print("Negative") 
    else : 
        print("Neutral")
        

sentiment(biden_speech)
sentiment(trump_speech)
sentiment(clinton16_speech)
sentiment(trump16_speech)
sentiment(harris20_speech)
sentiment(pence20_speech)
################################################################################################
# Sentiment Histograms

def sentence_sentiments(speech,color,title):
    sid = SentimentIntensityAnalyzer() 
    sentiment_list = list()
    sentence_list = nltk.tokenize.sent_tokenize(speech)
    for sentence in sentence_list:
        ss = sid.polarity_scores(sentence)
        sentiment_list.append(ss['compound'])
    plt.hist(sentiment_list, bins=15, color=color)
    plt.title(title)
    plt.xlabel('Sentiment Score')
    plt.ylabel('Frequency')
    plt.show()
    
sentence_sentiments(trump_speech, 'r', 'RNC 2020 Trump Speech Sentiment Scores')
sentence_sentiments(biden_speech, 'b', 'DNC 2020 Biden Speech Sentiment Scores')
sentence_sentiments(trump16_speech, 'r', 'RNC 2016 Trump Speech Sentiment Scores')
sentence_sentiments(clinton16_speech, 'b', 'DNC 2016 Clinton Speech Sentiment Scores')
sentence_sentiments(pence20_speech, 'r', 'RNC 2020 Pence Speech Sentiment Scores')
sentence_sentiments(harris20_speech, 'b', 'DNC 2020 Harris Speech Sentiment Scores')

################################################################################################
# Sentiment Central Tendency

def sentiment_ct(speech, pol):
    sid = SentimentIntensityAnalyzer()
    sentiment_list = list()
    sentence_list = nltk.tokenize.sent_tokenize(speech)
    for sentence in sentence_list:
        ss = sid.polarity_scores(sentence)
        sentiment_list.append(ss['compound'])
        print(pol, np.mean(np.array(sentiment_list)))
        print(len(sentiment_list))


trump20_sent = sentiment_ct(trump_speech, "Trump 2020 ")
biden20_sent = sentiment_ct(biden_speech, "Biden 2020 ")
pence20_sent = sentiment_ct(pence20_speech, "Trump 2020 ")
harris20_sent = sentiment_ct(harris20_speech, "Biden 2020 ")
trump16_sent = sentiment_ct(trump16_speech, "Trump 2016 ")
clinton16_sent = sentiment_ct(clinton16_speech, "Clinton 2016 ")



################################################################################################
# Bigrams
# 2020 POTUS
dnc_finder = BigramCollocationFinder.from_words(biden_tokens)
dnc_finder.nbest(BigramAssocMeasures.chi_sq, 30) # top 30 DNC bigrams
dnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30] # bigrams with scores
# plot barchart
plot_word_freqs(dnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], 'b', "Top 30 bigrams in Biden's 2020 DNC Speech", "Frequency Score")
# plot network
visualize_bigram(dnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], .6, "Top 30 bigrams in Biden's 2020 DNC Speech") # democrat network


rnc_finder = BigramCollocationFinder.from_words(trump_tokens)
rnc_finder.nbest(BigramAssocMeasures.raw_freq, 30) # top 30 RNC bigrams
rnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30] # bigrams with scores
# plot barchart
plot_word_freqs(rnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], 'r', "Top 30 bigrams in Trump's 2020 RNC Speech", "Frequency Score")
# plot network
visualize_bigram(rnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], 0.6, "Top 30 bigrams in Trump's 2020 RNC Speech") # republican network


# 2020 VPOTUS
dncvp_finder = BigramCollocationFinder.from_words(harris20_tokens)
dncvp_finder.nbest(BigramAssocMeasures.chi_sq, 30) # top 30 DNC bigrams
dncvp_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30] # bigrams with scores
# plot barchart
plot_word_freqs(dncvp_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], 'b', "Top 30 bigrams in Harris' 2020 DNC Speech", "Frequency Score")
# plot network
visualize_bigram(dncvp_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], .6, "Top 30 bigrams in Harris' 2020 DNC Speech") # democrat network


rncvp_finder = BigramCollocationFinder.from_words(pence20_tokens)
rncvp_finder.nbest(BigramAssocMeasures.raw_freq, 30) # top 30 RNC bigrams
rncvp_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30] # bigrams with scores
# plot barchart
plot_word_freqs(rncvp_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], 'r', "Top 30 bigrams in Pence's 2020 RNC Speech", "Frequency Score")
# plot network
visualize_bigram(rncvp_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], 0.6, "Top 30 bigrams in Pence's 2020 RNC Speech") # republican network


# 2016 POTUS
dnc16_finder = BigramCollocationFinder.from_words(clinton16_tokens)
dnc16_finder.nbest(BigramAssocMeasures.chi_sq, 30) # top 30 DNC bigrams
dnc16_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30] # bigrams with scores
# plot barchart
plot_word_freqs(dnc16_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], 'b', "Top 30 bigrams in Clinton's 2016 DNC Speech", "Frequency Score")
# plot network
visualize_bigram(dnc16_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], .6, "Top 30 bigrams in Clinton's 2016 DNC Speech") # democrat network


rnc16_finder = BigramCollocationFinder.from_words(trump16_tokens)
rnc16_finder.nbest(BigramAssocMeasures.raw_freq, 30) # top 30 RNC bigrams
rnc16_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30] # bigrams with scores
# plot barchart
plot_word_freqs(rnc16_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], 'r', "Top 30 bigrams in Trump's 2016 RNC Speech", "Frequency Score")
# plot network
visualize_bigram(rnc16_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], 0.6, "Top 30 bigrams in Trump's 2016 RNC Speech") # republican network







