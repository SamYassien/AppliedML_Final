import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.title(':newspaper: Fake News Detector')
st.markdown(
    'This app is a machine learning model that classifies news articles as real or fake.')

st.markdown(
    'The model was trained on a dataset of labeled news articles that can be found\
        [here](https://www.kaggle.com/datasets/jillanisofttech/fake-or-real-news?select=fake_or_real_news.csv).\
        The predictor variables used in the model are the title and the text of the article.')
st.markdown(
    'The model uses a passive-aggressive classifier. It achieves an accuracy of 94% on the test data when predicting the title & text,\
        and 72% when predicting the title only.')

st.markdown(
    'Use the form below to predict a news article\'s label. You can either predict both the title and text or just one of them.')

st.markdown('')

model = pickle.load(open('model/PAC_model.sav', 'rb'))

form = st.form(key='article_form')
article_title = form.text_input('Enter the title of the news article:')


use_text = st.checkbox(
    'Include the article text? (it would significantly improve the prediction accuracy)')

article_text = None

if use_text:
    article_text = form.text_area('Paste the news article text here:')

pred_button = form.form_submit_button(label='Predict')


def predict(title, text):

    if title:
        full_text = title
        if text:
            full_text = full_text + ' ' + text
    else:
        if text:
            full_text = text
        else:
            return None

    pred = model.predict([full_text])
    if pred[0] == 0:
        pred = 'Fake'
    else:
        pred = 'Real'

    return pred


if pred_button:
    if not article_title and not article_text:
        st.warning('Please enter the title and/or the text of the news article.')
    pred = predict(article_title, article_text)

    if pred is not None:
        st.markdown('##### Prediction')
        if pred == 'Fake':
            st.error('The news article is: Fake!')
        else:
            st.success('The news article is: Real!')
