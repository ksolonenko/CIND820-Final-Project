#importing libraries and packages for importing and analyizing dataset

import pandas as pd

#importing dataset as dataframe using pandas

data = pd.read_csv('IMDB_Dataset.csv')
    
#what the dataset looks like
data.head()

#number of unique values
data['sentiment'].value_counts()

#summary
data.describe()


#TEXT PROCESSING
import nltk

#separate by words
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()


#remove stop words
from nltk.corpus import stopwords
nltk.download('stopwords')
stopwords_list = nltk.corpus.stopwords.words('english')

def remove_stopwords(text):
    text=text.lower()
    data_tokens = text.apply(tokenizer)
    text = [token.strip() for token in data_tokens]
    return text

#removing noise
import re
def remove_special_chars(text):
    text=re.sub(r'[^a-zA-z0-9\s]','',text)
    return text

data['review'] = data['review'].apply(remove_special_chars)

data.head()


#text stemming
from nltk.stem.porter import PorterStemmer
def nltk_stemmer(text):
    stemmer=nltk.porter.PorterStemmer()
    text= ' '.join([stemmer.stem(word) for word in text.split()])
    return text

data['review'] = data['review'].apply(nltk_stemmer)


#BAG OF WORDS MODEL
from sklearn.feature_extraction.text import CountVectorizer



