# This Python 3 environment comes with many helpful analytics libraries installed
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))
        
        //IMPORTING LIBRARIES
        
        import numpy as np
import pandas as pd
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from nltk.stem.porter import PorterStemmer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

nltk.download('stopwords')
# printing the stopwords in English
print(stopwords.words('english'))

//PRE PROCESSING OF DATA

news_data = pd.read_csv('/content/train.csv')
news_data.head()

news_data.shape
# counting the number of missing values in the dataset
news_data.isnull().sum()
# replacing the null values with empty string
news_data = news_data.fillna('')
# checking the number of missing values in the dataset
news_data.isnull().sum()
# merging the author name and news title
news_data['content'] = news_data['author']+' '+news_data['title']
print(news_data['content'])
# separating the data & label

## Get the Independent Features
X = news_data.drop(columns='label', axis=1)
## Get the Dependent features
Y = news_data['label']
Y.value_counts()
X.shape
Y.shape
print(X)
print(Y)
port_stem = PorterStemmer()
def stemming(content):
    review = re.sub('[^a-zA-Z]',' ',content)
    review = review.lower()
    review = review.split()
    review = [port_stem.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    return review
    news_data['content'] = news_data['content'].apply(stemming)
    print(news_data['content'])
    #separating the data and label
X = news_data['content'].values
Y = news_data['label'].values
print(X)
print(Y)
Y.shape
# converting the textual data to numerical data
vectorizer = TfidfVectorizer()
vectorizer.fit(X)
X = vectorizer.transform(X)
print(X)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.2, stratify=Y, random_state=2)
