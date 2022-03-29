Importing  and describing data


```python
#importing dataset as dataframe using pandas

import pandas as pd
data = pd.read_csv('IMDB_Dataset.csv')
```


```python
#what the dataset looks like
data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>One of the other reviewers has mentioned that ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>A wonderful little production. &lt;br /&gt;&lt;br /&gt;The...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>I thought this was a wonderful way to spend ti...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Basically there's a family where a little boy ...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Petter Mattei's "Love in the Time of Money" is...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
#number of unique values
data['sentiment'].value_counts()
```




    positive    25000
    negative    25000
    Name: sentiment, dtype: int64




```python
#summary
data.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>50000</td>
      <td>50000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>49582</td>
      <td>2</td>
    </tr>
    <tr>
      <th>top</th>
      <td>Loved today's show!!! It was a variety and not...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>5</td>
      <td>25000</td>
    </tr>
  </tbody>
</table>
</div>




```python
print(data.shape)
```

    (50000, 2)
    

Text Processing


```python
import nltk

#separate by words
from nltk.tokenize.toktok import ToktokTokenizer
tokenizer = ToktokTokenizer()
```


```python
#remove stop words
from nltk.corpus import stopwords
stopwords_list = set(stopwords.words('english'))

def remove_stopwords(text):
    tokens = tokenizer.tokenize(text)
    tokens = [token.strip() for token in tokens]
    filtered_tokens = [token for token in tokens if token.lower() not in stopwords_list]
    filtered_text = ' '.join(filtered_tokens)    
    return filtered_text


data['review']=data['review'].apply(remove_stopwords)

data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>One reviewers mentioned watching 1 Oz episode ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wonderful little production. &lt;br / &gt;&lt;br / &gt;The...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>thought wonderful way spend time hot summer we...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Basically ' family little boy ( Jake ) thinks ...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Petter Mattei ' " Love Time Money " visually s...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
#removing noise
import re
def remove_special_chars(text):
    text=text.lower()
    text=re.sub(r'[^a-zA-z0-9\s]','',text)
    return text

data['review'] = data['review'].apply(remove_special_chars)

data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one review mention watch 1 oz episod hook righ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wonder littl product br br the film techniqu u...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>thought wonder way spend time hot summer weeke...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>basic famili littl boy jake think zombi closet...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>petter mattei love time money visual stun film...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>




```python
#text stemming
from nltk.stem.porter import PorterStemmer
def nltk_stemmer(text):
    stemmer=nltk.porter.PorterStemmer()
    text= ' '.join([stemmer.stem(word) for word in text.split()])
    return text

data['review'] = data['review'].apply(nltk_stemmer)

data.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>review</th>
      <th>sentiment</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>one review mention watch 1 oz episod hook righ...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>1</th>
      <td>wonder littl product br br the film techniqu u...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>2</th>
      <td>thought wonder way spend time hot summer weeke...</td>
      <td>positive</td>
    </tr>
    <tr>
      <th>3</th>
      <td>basic famili littl boy jake think zombi closet...</td>
      <td>negative</td>
    </tr>
    <tr>
      <th>4</th>
      <td>petter mattei love time money visual stun film...</td>
      <td>positive</td>
    </tr>
  </tbody>
</table>
</div>



BAG OF WORDS MODEL 

The Bag of Words Model converts text into numerical representation, so that text data can be used to train models.
This model uses word tokens from the entire set of data.


```python
#split into 80% train and 20% test set

train_data = data.sample(frac=0.8, random_state=25)
test_data = data.drop(train_data.index)

print(train_data.shape)
print(test_data.shape)
```

    (40000, 2)
    (10000, 2)
    

Naive Bayes Model


```python
#split data into train and test sets
from sklearn.model_selection import train_test_split as tts

x = data.iloc[0:,0].values
y = data.iloc[0:,1].values

x_train, x_test, y_train, y_test = tts(x, y, test_size=0.2, random_state=5555)

```


```python
#Naive Bayes model
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()

#importing Naive Bayes
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

bayes = MultinomialNB()

model=Pipeline([('vectorizer',tf),('classifier',bayes)])

model.fit(x_train, y_train)
```




    Pipeline(steps=[('vectorizer', TfidfVectorizer()),
                    ('classifier', MultinomialNB())])



Model Testing and Results


```python
#model prediction
y_pred = model.predict(x_test)
```


```python
#model results

from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(y_pred,y_test)

```




    0.8655




```python
#confusion matrix

cmatrix =confusion_matrix(y_test,y_pred)
print(cmatrix)
```

    [[4381  626]
     [ 719 4274]]
    


```python
#recall and precision


from sklearn.metrics import classification_report
nbmodel_results= classification_report(y_test,y_pred,target_names=['Positive','Negative'])
print(nbmodel_results)
```

                  precision    recall  f1-score   support
    
        Positive       0.86      0.87      0.87      5007
        Negative       0.87      0.86      0.86      4993
    
        accuracy                           0.87     10000
       macro avg       0.87      0.87      0.87     10000
    weighted avg       0.87      0.87      0.87     10000
    
    

Decision Tree Model


```python
#Decision Tree model
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
tf = TfidfVectorizer()

#importing Naive Bayes
from sklearn.pipeline import Pipeline
from sklearn import tree 

clf = tree.DecisionTreeClassifier()

treemodel=Pipeline([('vectorizer',tf),('classifier',clf)])

treemodel.fit(x_train, y_train)

```




    Pipeline(steps=[('vectorizer', TfidfVectorizer()),
                    ('classifier', DecisionTreeClassifier())])



Model Testing and Results


```python
#model prediction
tree_pred = treemodel.predict(x_test)
```


```python
#model results

from sklearn.metrics import accuracy_score, confusion_matrix
accuracy_score(tree_pred,y_test)

```




    0.7168




```python
#confusion matrix

cmatrix =confusion_matrix(y_test,tree_pred)
print(cmatrix)
```

    [[3593 1414]
     [1418 3575]]
    


```python
#recall and precision

from sklearn.metrics import classification_report
nbmodel_results= classification_report(y_test,tree_pred,target_names=['Positive','Negative'])
print(nbmodel_results)
```

                  precision    recall  f1-score   support
    
        Positive       0.72      0.72      0.72      5007
        Negative       0.72      0.72      0.72      4993
    
        accuracy                           0.72     10000
       macro avg       0.72      0.72      0.72     10000
    weighted avg       0.72      0.72      0.72     10000
    
    
