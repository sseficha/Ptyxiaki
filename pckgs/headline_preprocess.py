from datetime import datetime
import re
import string
from gensim.models.phrases import Phrases
from numpy import nan
import pandas as pd
from pckgs.helper import timeseries_to_supervised, timeseries_to_supervised2
from nltk.tokenize import word_tokenize


# import nltk
# nltk.download('wordnet')
# from nltk.corpus import stopwords
# from nltk.stem import WordNetLemmatizer


class HeadlinePreprocess:

    @staticmethod
    def preprocess(df, bigrams=False, tokenize=False):
        df.rename(columns={'publishdate': 'date'}, inplace=True)
        df.rename(columns={'headlinetext': 'text'}, inplace=True)
        df.date = df.date.map(lambda p: datetime.strptime(str(p), '%Y%m%d'))
        # to lowercase
        df.text = df.text.map(lambda p: p.lower())
        # remove number
        # df.text = df.text.map(lambda p: re.sub(r'\d+', '', p))
        # remove punctuation
        df.text = df.text.map(lambda p: p.translate(str.maketrans("", "", string.punctuation)))
        # remove whitespace
        df.text = df.text.map(lambda p: p.strip())
        # tokenize
        if tokenize:
            df.text = df.text.map(lambda text: [word for word in word_tokenize(text)])
        # get bigrams
        if bigrams:
            bigram = Phrases(df.text.to_numpy(), min_count=30, progress_per=10000)
            df.text = df.text.map(lambda p: bigram[p])


        # # drop those with length <2
        # df.text = df.text.map(lambda p: nan if len(p) < 2 else p)
        # df.dropna(inplace=True)
        # df = df.reset_index(drop=True)

        # #remove stop words
        # stop_words = set(stopwords.words('english'))
        # df.text = df.text.map(lambda text : [word for word in word_tokenize(text) if not word in stop_words])
        # del stop_words
        # #lemmatization
        # lemmatizer=WordNetLemmatizer()
        # df.text = df.text.map(lambda text : [lemmatizer.lemmatize(word) for word in text])
        return df

    @staticmethod
    def shape_vectors(df, lag, index):
        # shifted = pd.DataFrame(index=df.index)
        # for i in range(len(df.columns)):
        #     temp = timeseries_to_supervised(df, str(i), lag)
        #     temp.drop(str(i) + '_t', axis=1, inplace=True)  # drop same day
        #     shifted = pd.concat([shifted, temp], axis=1)

        shifted = timeseries_to_supervised2(df, lag)
        shifted.drop(shifted.iloc[:,:768].columns, axis=1, inplace=True)

        shifted.dropna(inplace=True)
        shifted = shifted.reindex(index)
        print(shifted.head())
        shifted = shifted.to_numpy()
        # shifted = shifted.reshape(shifted.shape[0], lag - 1, int(shifted.shape[1] / (lag - 1)), order='F')      #change order with timeseresi2222!!!!
        shifted = shifted.reshape(shifted.shape[0], lag - 1, int(shifted.shape[1] / (lag - 1)))      #change order with timeseresi2222!!!!
        print(shifted.shape)
        return shifted

