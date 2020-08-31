import pymongo
import nltk
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import re
import numpy as np
from wordcloud import WordCloud 
import matplotlib.pyplot as plt
import networkx as nx
client = pymongo.MongoClient('localhost',27017)
#list the databases defined
client.database_names()

db = client.political_tweets
db.collection_names()

rnc_coll=db.rnc_tweets
#rnc_coll.drop()
rnc_docs = rnc_coll.find()

dnc_coll = db.dnc_tweets
#dnc_coll.drop()
dnc_docs = dnc_coll.find()


# convert the document cursor to a list of documents
rnc_doclist = [tweet['text'] for tweet in rnc_docs]
dnc_doclist = [tweet['text'] for tweet in dnc_docs]
len(rnc_doclist)
len(dnc_doclist)

# make sure all tweets are unique
rnc_array = np.array(rnc_doclist)
len(np.unique(rnc_array))
dnc_array = np.array(dnc_doclist)
len(np.unique(dnc_array))


print(rnc_doclist[:1])

dnc_tokens = list()
rnc_tokens = list()

ttokenizer = nltk.tokenize.TweetTokenizer()
nltk_stopwords = nltk.corpus.stopwords.words('english')
nltk_stopwords.extend(['rt','https','co','2020'])
def tweet_to_tokens(tweet):
    s = tweet.lower()
    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^@a-zA-Z0-9\s]', ' ', s)
    tokens = ttokenizer.tokenize(s)
    filtered_tokens = [w for w in tokens if w not in nltk_stopwords]
    return filtered_tokens

for tweet in dnc_doclist:
    for token in tweet_to_tokens(tweet):
        dnc_tokens.append(token)

for tweet in rnc_doclist:
    for token in tweet_to_tokens(tweet):
        rnc_tokens.append(token)

print(rnc_tokens[:25])

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

dnc_msgFD = nltk.FreqDist(dnc_tokens)
dnc_top_words = dnc_msgFD.most_common(30)
# horizontal bar chart
plot_word_freqs(dnc_top_words, 'b', "Top 30 words in @DNCConvention2020")
# wordcloud
wc_dnc = WordCloud(background_color="white",width=1000,height=1000, max_words=30,relative_scaling=0,normalize_plurals=False).generate_from_frequencies(dict(dnc_top_words))
plt.imshow(wc_dnc)

rnc_msgFD = nltk.FreqDist(rnc_tokens)
rnc_top_words = rnc_msgFD.most_common(30)
# bar chart
plot_word_freqs(rnc_top_words, 'r', "Top 30 words in @RNCConvention2020")
# wordcloud
wc_rnc = WordCloud(background_color="white",width=1000,height=1000, max_words=30,relative_scaling=0,normalize_plurals=False).generate_from_frequencies(dict(rnc_top_words[1:]))
plt.imshow(wc_rnc)


def tweet_to_tokens_with_stopwords(tweet):
    s = tweet.lower()
    # Replace all none alphanumeric characters with spaces
    s = re.sub(r'[^@a-zA-Z0-9\s]', ' ', s)
    tokens = ttokenizer.tokenize(s)
    #filtered_tokens = [w for w in tokens if w not in nltk_stopwords]
    return tokens

####################################################
# Bigram Analysis

dnc_tokens_by_document = list()
rnc_tokens_by_document = list()
for tweet in dnc_doclist:
    dnc_tokens_by_document.append(tweet_to_tokens(tweet))

for tweet in rnc_doclist:
    rnc_tokens_by_document.append(tweet_to_tokens(tweet))

dnc_finder = BigramCollocationFinder.from_documents(dnc_tokens_by_document)
dnc_finder.nbest(BigramAssocMeasures.raw_freq, 30) # top 30 DNC bigrams
dnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30] # bigrams with scores

dict(dnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30])

rnc_finder = BigramCollocationFinder.from_documents(rnc_tokens_by_document)
rnc_finder.nbest(BigramAssocMeasures.raw_freq, 30) # top 30 RNC bigrams
rnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30] # bigrams with scores

# function to create visual representation of bigrams as a network of nodes
def visualize_bigram(list_of_bigram_tuples, kval):
    bigram_dict = dict(list_of_bigram_tuples)
    # Create network plot 
    G = nx.Graph()
    
    # Create connections between nodes
    for k, v in bigram_dict.items():
        G.add_edge(k[0], k[1], weight=(v * 10))
    
    fig, ax = plt.subplots(figsize=(15, 12))
    pos = nx.spring_layout(G, k=kval) # k is used to adjust distance between nodes
    # Plot networks
    nx.draw_networkx(G, pos,
                     font_size=18,
                     width=2,
                     edge_color='grey',
                     node_color='purple',
                     with_labels = False,
                     ax=ax)
    
    # Create offset labels
    for key, value in pos.items():
        x, y = value[0]+.05, value[1]+.05
        ax.text(x, y,
                s=key,
                bbox=dict(facecolor='red', alpha=0.25),
                horizontalalignment='center', fontsize=13)
        
    plt.show()
    
visualize_bigram(dnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], .2) # DNC network
visualize_bigram(rnc_finder.score_ngrams(BigramAssocMeasures.raw_freq)[:30], .2) # RNC network
