# Model Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import keras
#from keras.utils import pad_sequences
from keras.models import Sequential
from keras.layers import Embedding, Flatten, Dense, SimpleRNN, LSTM, GRU, Dropout
from sklearn.metrics import confusion_matrix
from seaborn import heatmap
#from gensim.models.keyedvectors import KeyedVectors
import tensorflow as tf
from sklearn.model_selection import train_test_split
import nltk
from os import getcwd
#import gensim
#from gensim.test.utils import common_texts
import re
import string
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import TweetTokenizer
from keras.models import load_model
from sklearn.metrics import classification_report
import pickle
import requests
from bs4 import BeautifulSoup
from unicodedata import normalize
import re
import xgboost as xgb

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


def limpieza_str(s):
  s = re.sub(r"([^n\u0300-\u036f]|n(?!\u0303(?![\u0300-\u036f])))[\u0300-\u036f]+", r"\1", 
          normalize( "NFD", s), 0, re.I)
  s = normalize( 'NFC', s).lower().replace('\n','')
  return(s.lstrip().rstrip())

def mrep(s):
  repl = {',':'.',';':'.','-':'.',':':'.'}
  for k, i in repl.items():
    s=s.replace(k,i)
  return s

def word_gender(word):
    if word.upper()[-1] == 'O':
        gender = 'M'
    elif word.upper()[-1] == 'A':
        gender = 'F'
    else:
        gender = 'N'
    return gender


def bias_model_prediction(url):
    
    ## url data
    headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 6.1; WOW64; rv:50.0) Gecko/20100101 Firefox/50.0'}
    r = requests.get(url, headers=headers)
    soup = BeautifulSoup(r.text)
    
    ## extracción
    busq = ['p','h1','h2','h3','h4','h5','h6','menu','title','dt','hr','li','ol','ul','caption','tr']
    Textos = {finds: soup.find_all(finds) for finds in busq if len(soup.find_all(finds))!=0}
    textos = {keys:[limpieza_str(ele.text) for ele in Textos[keys]] for keys in Textos.keys()}
    # Si no se puede usar limpieza_str
    # textos = {keys:[ele.text.strip().lower() for ele in Textos[keys]] for keys in Textos.keys()}

    frases = [item for k, sublist in textos.items() for item in sublist]
    spliteo = [mrep(f).split('.') for f in frases]
    # Que tenga al menos 3 palabras
    frases_p = list(set([item.strip().replace('  ',' ') for sublist in spliteo for item in sublist if item.strip().count(' ')>2]))
    
    ## array preprocess
    X_raw=np.array(frases_p)
    preprocess_list = np.array([process_tweet(x) for x in X_raw])
    
    ## gender list
    gender_list = []
    for frase in X_raw:
        for word in frase:
            gender_list.append(word_gender(word))
    masc, fem = (pd.Series(gender_list).value_counts(1)['F'], pd.Series(gender_list).value_counts(1)['M'])
    
    if masc > fem:
        incli = 'MASCULINO'
    else:
        incli = 'FEMENINO'
    
    # Word to vector
    maxlen = 100 #max number of word
    max_words = 20000 #considers the first 20000 words
    

    ## TOKENIZER
    #myfile = drive.CreateFile({'id': '1pOJ_u8rHxndklavjBgUSWzlLuP2aWoKO'})
    #myfile.GetContentFile('tokenizer_bbva3.pickle')
    file = open('/home/edco17/Escritorio/hackaton2022/tokenizer_bbva3.pickle', 'rb')
    tokenizer = pickle.load(file)
    sequences = tokenizer.texts_to_sequences(preprocess_list)
    
    # Word idctionary
    word_index = tokenizer.word_index
    data = pad_sequences(sequences, maxlen=maxlen)
    
    ## model
    #myfile = drive.CreateFile({'id': '1FxlOTsTLdyW6evDzUGQLyjIgwNebg6M6'})
    #myfile.GetContentFile('model_Hackathon_22.h5')
    model = xgb.Booster()
    model.load_model("/home/edco17/Escritorio/hackaton2022/xg_model_Hackathon.txt")
    #model = load_model('/home/edco17/Escritorio/hackaton2022/xg_model_Hackathon.h5')
    #model = load_model('/home/edco17/Escritorio/hackaton2022/model_light_Hackathon.h5')

    ## prediction
    #prediction = model.predict(xgb.DMatrix(data))
    prediction = (model.predict(xgb.DMatrix(data))>0.43).astype('int')
    df_pred = pd.DataFrame(frases_p, columns = ['full_text'])
    df_pred['prediction'] = prediction
    
    ## JSON 
    resultado = {}
    resultado['porcentaje_sesgo'] = prediction.mean()
    resultado['proporcion_masculina'] = masc
    resultado['proporcion_femenina'] = fem
    resultado['inclinacion'] = incli
    
    ## model return
    return resultado
