#!pip install tensorflow==2.8

## libraries

# Model Libraries
import numpy as np
import pandas as pd
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
#from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, SimpleRNN, LSTM, GRU, Dropout

import tensorflow as tf
import nltk
from os import getcwd

import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from keras.models import load_model
import pickle
import boto3


def process_tweet(tweet):
    """Process tweet function.
    Input:
        tweet: a string containing a tweet
    Output:
        tweets_clean: a list of words containing the processed tweet

    """
    stemmer = PorterStemmer()
    stopwords_english = stopwords.words('spanish')
    # remove stock market tickers like $GE
    tweet = re.sub(r'\$\w*', '', tweet)
    # remove old style retweet text "RT"
    tweet = re.sub(r'^RT[\s]+', '', tweet)
    # remove hyperlinks
    tweet = re.sub(r'https?:\/\/.*[\r\n]*', '', tweet)
    # remove hashtags
    # only removing the hash # sign from the word
    tweet = re.sub(r'#', '', tweet)
    # tokenize tweets
    tokenizer = TweetTokenizer(preserve_case=False, strip_handles=True,
                               reduce_len=True)
    tweet_tokens = tokenizer.tokenize(tweet)

    tweets_clean = []
    for word in tweet_tokens:
        if (word not in stopwords_english and  # remove stopwords
                word not in string.punctuation):  # remove punctuation
            # tweets_clean.append(word)
            stem_word = stemmer.stem(word)  # stemming word
            tweets_clean.append(stem_word)

    return tweets_clean


def build_freqs(tweets, ys):
    """Build frequencies.
    Input:
        tweets: a list of tweets
        ys: an m x 1 array with the sentiment label of each tweet
            (either 0 or 1)
    Output:
        freqs: a dictionary mapping each (word, sentiment) pair to its
        frequency
    """
    # Convert np array to list since zip needs an iterable.
    # The squeeze is necessary or the list ends up with one element.
    # Also note that this is just a NOP if ys is already a list.
    yslist = np.squeeze(ys).tolist()

    # Start with an empty dictionary and populate it by looping over all tweets
    # and over all processed words in each tweet.
    freqs = {}
    for y, tweet in zip(yslist, tweets):
        for word in process_tweet(tweet):
            pair = (word, y)
            if pair in freqs:
                freqs[pair] += 1
            else:
                freqs[pair] = 1

    return freqs

nltk.download('twitter_samples')
nltk.download('stopwords')



def bias_model_prediction(df):
    ## array preprocess
    X_raw=np.array(df)
    preprocess_list = np.array([process_tweet(x) for x in X_raw])
    
    # Word to vector
    maxlen = 100 #max number of word
    max_words = 20000 #considers the first 20000 words

    ## TOKENIZER
    s3 = boto3.resource('s3')
    file = pickle.loads(s3.Bucket("hackaton22/Auxiliares/").Object("tokenizer_bbva3.pickle").get()['Body'].read())
    #file = open('D:\\Users\\Ian\\Documents\\BBVA2022\\tokenizer_bbva3.pickle', 'rb')
    tokenizer = pickle.load(file)
    sequences = tokenizer.texts_to_sequences(preprocess_list)

    # Word idctionary
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=maxlen)
    
    ## model
    s3_client = boto3.client('s3')
    s3_client.download_file('hackaton22/Model/','model_Hackathon_22.h5','model') #downloading the model to temporary file named "model"
    with h5py.File('model','r') as f:
        model = load_model(f)
    #model = load_model('D:\\Users\\Ian\\Downloads\\model_Hackathon_22.h5')

    ## prediction
    prediction = (model.predict(data)[:,1]>0.49).astype('int')
    df_pred = pd.DataFrame(df, columns = ['full_text'])
    df_pred['prediction'] = prediction
    
    ## model return
    return df_pred

def lambda_handler(event, context):
    frases = pd.Series(event['full_text'])
    print(frases[0])
    
    df_pred = bias_model_prediction(frases)
    
    transactionResponse = {}
    transactionResponse['full_text'] = df_pred['full_text']
    transactionResponse['prediction'] = df_pred['prediction']
    
    responseObject = {}
    responseObject['statusCode'] = 200
    responseObject['headers'] = {}
    responseObject['headers']['Content-Type'] = 'application/json'
    responseObject['body'] = json.dumps(transactionResponse)
    
    # TODO implement
    return {responseObject}
