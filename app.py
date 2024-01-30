import streamlit as st
st.set_page_config(layout="wide")
st.title('Sentiment Analysis')
st.write("##### Type something to check the sentiment")

#input
text = st.text_input('Text')
pressed = st.button('Submit')

from nltk.sentiment import SentimentIntensityAnalyzer
sia = SentimentIntensityAnalyzer()


from transformers import pipeline
model_id = "distilbert-base-uncased-finetuned-sst-2-english"
sent_pipeline = pipeline("sentiment-analysis",model=model_id)


# Roberta Pretrained Model
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification
from scipy.special import softmax

MODEL = f"cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)


#for roberta
encoded_text = tokenizer(text, return_tensors='pt')
output = model(**encoded_text)
scores = output[0][0].detach().numpy()
scores = softmax(scores)
scores_dict = {
    'roberta_neg' : scores[0],
    'roberta_neu' : scores[1],
    'roberta_pos' : scores[2]
}

res = sia.polarity_scores(text)
col1, col2, col3 = st.columns(3)
if(pressed):
    col1.bar_chart(data=res)
    col2.bar_chart(sent_pipeline(text))
    col3.bar_chart(data = scores_dict)
else:
    st.write()