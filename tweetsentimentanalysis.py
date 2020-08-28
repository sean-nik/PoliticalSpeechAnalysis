import pymongo
import nltk
import numpy as np
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
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


dnc_sentiment_list = list()
rnc_sentiment_list = list()

sid = SentimentIntensityAnalyzer()
for tweet in rnc_doclist:
    ss = sid.polarity_scores(tweet)
    rnc_sentiment_list.append(ss['compound'])
len(rnc_sentiment_list)
rnc_sentiment_list[:10]
rnc_avg_sentiment = np.mean(np.array(rnc_sentiment_list))
#0.1260588
rnc_stddev_sentiment = np.std(np.array(rnc_sentiment_list))
#0.43161457012311344
plt.hist(rnc_sentiment_list, bins=15, color='red')
plt.title("#RNCConvention2020 Tweet Sentiment Scores")
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()


for tweet in dnc_doclist:
    ss = sid.polarity_scores(tweet)
    dnc_sentiment_list.append(ss['compound'])
len(dnc_sentiment_list)
dnc_sentiment_list[:10]
dnc_avg_sentiment = np.mean(np.array(dnc_sentiment_list))
#0.0230778
dnc_stddev_sentiment = np.std(np.array(dnc_sentiment_list))
#0.40934259549570456
plt.hist(dnc_sentiment_list, bins=15, color='blue')
plt.title("#DNCConvention2020 Tweet Sentiment Scores")
plt.xlabel('Sentiment Score')
plt.ylabel('Frequency')
plt.show()
