import streamlit as st


st.title('Sentiment Analysis')
st.write("##### Type something to check the sentiment")

from nltk.sentiment import SentimentIntensityAnalyzer

sia = SentimentIntensityAnalyzer()

text = st.text_input('Text')
pressed = st.button('Submit')

res = sia.polarity_scores(text)

if(pressed):
    st.write(res)
    st.bar_chart(data=res)
else:
    st.write('Press submit button')