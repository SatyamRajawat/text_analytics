from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
import spacy
from collections import Counter
import string
import seaborn as sns
import os
import time
import re
import gensim
from gensim import models,corpora
punct = string.punctuation
from spacytextblob.spacytextblob import SpacyTextBlob
spacy_text_blob = SpacyTextBlob()
nlp = spacy.load('en_core_web_sm') #Loading spacy english
nlp.add_pipe(spacy_text_blob)
# Add Stop words
nlp.Defaults.stop_words |= {"hut","called","calling",}

import pyLDAvis
import pyLDAvis.gensim  # don't skip this
import matplotlib.pyplot as plt
from IPython.display import HTML

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning) 
import imp

app = Flask(__name__)


#  =============== Loading functions here ====================

# Text Processing(cleaning) 
def text_clean(text):
    text = text.lower()  #Convert text in lower case
    punc_removed = [char for char in text if char not in punct]  #Removing Punctuations
    punc_removed_join = ''.join(punc_removed) 
    
    doc= nlp(punc_removed_join)
    text_out = [token.lemma_ for token in doc if token.is_stop == False and token.is_alpha and len(token)>2]
    txt = ' '.join(text_out)
    return txt

# Create function for polarity checking
def polarity(text):
    doc = nlp(text)
    pol = float(format(doc._.sentiment.polarity, '.3f'))
    return pol

# Function for topic identification
def topic_token(text):
    removal=['ADV','PRON','CCONJ','PUNCT','PART','DET','ADP','SPACE']  #get Noun phrase
    text_out = []
    doc= nlp(text)
    for token in doc:
        if token.pos_ not in removal:
            lemma = token.lemma_            #lemmatization of token word
            text_out.append(lemma)            
    return text_out
# Function for getting emotion word_list
def emotion_analysis(text):
    emotion_list = []
    with open('model/Emotions.txt', 'r') as file:
        for line in file:
            clear_line = line.replace("\n", '').replace(",", '').replace("'", '').strip()
            word, emotion = clear_line.split(':')
        
            if word in text:
                emotion_list.append(emotion)
    return emotion_list



# ============================ Calling html page=======================================
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/analytics', methods=['POST'])
def form_post():
    text = request.form['txt']

    a_dict = {}
    print("Predicting model start ........")
    # ----------------- Sentiment Analyse by file--------------------

    if len(text) == 0:
        f = request.files['file']
        f.save(f.filename)
        df = pd.read_csv(f.filename, encoding='latin1')
        df['clean_doc'] = df['Text'].apply(text_clean)
        df['token'] = df['clean_doc'].apply(topic_token)

        df['polarity'] = df['clean_doc'].apply(polarity)
        df['emotion'] = df['clean_doc'].apply(emotion_analysis)
        df['emotion'] = df['emotion'].apply(lambda x:','.join(map(str, x)))
        df['sentiment'] = df['polarity'].apply(lambda x: 'Positive' if x > 0 else ('Negative' if x < 0 else 'Neutral'))
        
                
        DataFrame = pd.DataFrame({"text": df['Text'],'polarity':df['polarity'],'sentiment':df['sentiment'],'emotion':df['emotion']})
        # Sentiment_data = df.to_html(escape=False, index=True,)


        
        # ----------Topic Identification ----------------------
        

        val = []
        for i in range(len(df)):
            val.append(df['token'][i])
        
        dictionary = corpora.Dictionary(val)
        # dictionary

        bow_corpus = [dictionary.doc2bow(doc) for doc in val]   #Creating bag of words

        # process gensim model for topic identification
        lda_model =  gensim.models.LdaMulticore(bow_corpus, num_topics = 10, id2word = dictionary, passes = 10,workers = 2)

        topics = {}
        num_topics = 10
        for i in range(num_topics):
            tt = lda_model.get_topic_terms(i,10)
            topic = ', '.join([dictionary[pair[0]] for pair in tt])
            print("TOPIC: {} \nTOPIC WORDS : {}".format(i+1, topic ))
            print()
            topics[i] = topic


           
        vis = pyLDAvis.gensim.prepare(lda_model, bow_corpus, dictionary)
        pyLDAvis.save_html(vis,'templates/vis.html')
        # -------------------------------------
        return render_template('index.html',data=DataFrame.to_dict(orient='records'),topics=topics, output=True)

    else:
        
        text1 = text_clean(text)

        # Sentiment Analysis On text===============================

      
        polar = polarity(text1)
        
        def sent(polar):
            if polar > 0:
                p = "Positive"
            elif polar == 0:
                p = "Neutral"
            else :
                p = "Negative"
            return p
        sentiment = sent(polar)
        print(sentiment)
        Texts = text
        Polarity = polar
        Sentiment = sentiment
        
        # Emotion Analysis==================================
        emotion_list = emotion_analysis(text1)
        separator=', '
        st = separator.join(emotion_list)
            

        # ---------------Emotion Graph -----------
        try:
            emotion_word = Counter(emotion_list)        
            emotion_data = pd.DataFrame({'Words': list(emotion_word.keys()), 'Count': list(emotion_word.values())})
            plt.figure(figsize=(15, 5))
            emotion_data_graph1 = emotion_data.head(10)
            sns.barplot(x='Words', y='Count', data=emotion_data_graph1,color=("blue")).set_title('Emotion Visual With(Words & Counts)')
            Emotion_graph_name1 = "e1graph" + str(time.time()) + ".png"
            for filename in os.listdir('static/image/'):
                if filename.startswith('e1graph'):  # not to remove other images
                    os.remove('static/image/' + filename)

            plt.savefig('static/image/' + Emotion_graph_name1, bbox_inches='tight',dpi = 420)
    
        except ValueError:
                  Emotion_graph_name1 = "No"


        return render_template('index.html',Texts=Texts,Polarity=Polarity,Sentiment=Sentiment,emotion=st,
                                            Emotion_graph=Emotion_graph_name1, output1=True)


@app.route('/topic', methods=['GET'])
def html():
    return render_template('vis.html', out=True)



if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5001, threaded=True)
