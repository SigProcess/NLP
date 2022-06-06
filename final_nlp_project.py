# -*- coding: utf-8 -*-
"""Final_NLP_Project.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1jWJwTMNZEh7kh_yNFOgzOqaU06j5HGxq
"""
import pandas as pd
import pickle
#from nltk.corpus import stopwords
df=pd.read_excel(r'C:\Users\Tanushree Sharma\Desktop\1000leads.xlsx')
df.drop(df.columns[4],axis=1, inplace=True)
df.head()
df=df.dropna()
df.rename(columns={'Status ':'Status'}, inplace=True)
df.reset_index(inplace=True)
X=df['Status information']
y=df['Status'];print(type(y))
y=list(y)
print(y)
import regex as re
for i in range(len(y)):
  y[i]=re.sub('N.*', 'Not Converted', y[i])
  y[i]=re.sub('C.*', 'Converted', y[i])
y=pd.Series(y)
y.value_counts()
# Import label encoder
from sklearn import preprocessing

# label_encoder object knows how to understand word labels.
label_encoder = preprocessing.LabelEncoder()
  
# Encode labels in column 'species'.
y= label_encoder.fit_transform(y)
(X.shape,y.shape)
# Data cleaning
from nltk.stem.porter import PorterStemmer
ps = PorterStemmer()
corpus = []
for i in range(0, len(X)):
  
    review = re.sub(r'[^a-zA-Z]', ' ', X[i])
    review = review.lower()
    review = review.split()
    #review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
    review = ' '.join(review)
    corpus.append(review)

## TFidf Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
vectorizer=CountVectorizer(decode_error='replace')
vec_train=vectorizer.fit_transform(corpus)
pickle.dump(vectorizer.vocabulary_,open("feature.pkl","wb"))
transformer = TfidfTransformer()
X=transformer.fit_transform(vec_train).toarray()
#tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,4))
#X=tfidf_v.fit_transform(corpus).toarray()
X.shape

#Random Oversampling

from imblearn.over_sampling import RandomOverSampler
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
ros = RandomOverSampler(random_state=777)
X_ROS, y_ROS = ros.fit_resample(X, y)
X_train, x_test, Y_train, y_test = train_test_split(X_ROS,y_ROS,test_size=0.3,random_state=42)

#Using Multinomial NB
classifier=MultinomialNB()
classifier.fit(X_train, Y_train)
y_pred = classifier.predict(x_test)
print('Accuracy %s' % accuracy_score(y_pred,y_test))
print(classification_report(y_test,y_pred))

#Using Passive Aggressive Classifier
from sklearn import metrics
from sklearn.linear_model import PassiveAggressiveClassifier
linear_clf = PassiveAggressiveClassifier()
linear_clf.fit(X_train, Y_train)
y_pred = linear_clf.predict(x_test)
score = metrics.accuracy_score(y_test, y_pred)
print('Accuracy %s' % accuracy_score(y_pred,y_test))
print(classification_report(y_test,y_pred))

pickle_out = open("nlpmodel.pkl","wb")
pickle.dump(linear_clf, pickle_out)
pickle_out.close()