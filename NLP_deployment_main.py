# -*- coding: utf-8 -*-
"""
Created on Thu May 12 11:06:24 2022

@author: Tanushree Sharma
"""

import pickle
import regex as re
import numpy as np
#from flasgger import Swagger
import streamlit as st 
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
#app=Flask(__name__)
#Swagger(app)

pickle_in = open("nlpmodel.pkl","rb")
algo=pickle.load(pickle_in)



#@app.route('/')
def welcome():
    return "Welcome All"

def predict_conversion(message):
    
        review = re.sub(r'[^a-zA-Z]', ' ', message)
        review = review.lower()
        review = review.split()
        #review = [ps.stem(word) for word in review if not word in stopwords.words('english')]
        review = ' '.join(review)
        transformer = TfidfTransformer()

        loaded_vec = CountVectorizer(decode_error="replace",vocabulary=pickle.load(open("feature.pkl", "rb")))

        X = transformer.fit_transform(loaded_vec.fit_transform(np.array([review])))
        #tfidf_v=TfidfVectorizer(max_features=5000,ngram_range=(1,4), use_idf=True)
        
        #X=tfidf_v.fit_transform([review]).toarray()
        print(X)         
        prediction=algo.predict(X)
        if prediction>=0.5:
            prediction='Will not be converted'
        else:
            prediction='Will be converted'
            
        return prediction


def main():
    st.title("Lead Conversion prediction")
    html_temp = """
    <div style="background-color:tomato;padding:10px">
    <h2 style="color:white;text-align:center;">Streamlit Lead Conversion Prediction App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    message = st.text_input("message","Type Here")
      
    result=""
    if st.button("Predict"):
        result=predict_conversion(message)
    st.success('The output is {}'.format(result))
    if st.button("About"):
        st.text("Lets Learn")
        st.text("Built with Streamlit")

if __name__=='__main__':
    main()
    
