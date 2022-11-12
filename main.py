# This is a sample Python script.
from datetime import time

# Press Maj+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.
import streamlit as st
from PIL import Image
import random
import nltk
import pickle
import textblob
from textblob import TextBlob
import re
import contractions
import unidecode
from nltk import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
nltk.download('stopwords')


from nltk.stem.wordnet import WordNetLemmatizer

nltk.download('averaged_perceptron_tagger')
from nltk.corpus import wordnet

nltk.download('wordnet')
nltk.download('omw-1.4')

with open('./NMF_Model.pkl', 'rb') as NMF_Model:
    pkl1 = pickle.load(NMF_Model)
print(pkl1)
with open('./vectorizer_Model.pkl', 'rb') as vectorizer_Model:
    pkl2 = pickle.load(vectorizer_Model)
print(pkl2)

import nltk
nltk.download('punkt')
labels_dict = {
        0: "bad customer service at phone",
        1: "disappointed taste",
        2: "cold pizza",
        3: "not enough chicken",
        4: "bad food quality",
        5: "bad service",
        6: "bad burgers",
        7: "long wait",
        8: "bad experience on all levels",
        9: "bad experience at the bar",
        10: "problem with the delivery or the online order",
        11: "bad experience several times",
        12: "unorganized staff",
        13: "poor quality sushis",
        14: "dangerous place"
    }


DATASET_FILE = "./dataset_cleaned.csv"
dataset_df = pd.read_csv(DATASET_FILE)
Negative_data = dataset_df[(dataset_df['stars'] == 1) | (dataset_df['stars'] == 2)]
# having a datafarme containing negative data
Neg_DF = pd.DataFrame(columns=['text_cleaned','stars','length'])
Neg_DF.text_cleaned= Negative_data.text_cleaned
Neg_DF.stars= Negative_data.stars
Neg_DF.length= Negative_data.length
Neg_DF.index=range(len(Negative_data))
corpus = Neg_DF.text_cleaned
### Texte en minuscule
def lower_case_convertion(text):
	lower_text = text.lower()
	return lower_text

### Suppression des tags HTML
def remove_html_tags(text):
	html_pattern = r'<.*?>'
	without_html = re.sub(pattern=html_pattern, repl=' ', string=text)
	return without_html

# Suppression des URLS
def remove_urls(text):
	url_pattern = r'https?://\S+|www\.\S+'
	without_urls = re.sub(pattern=url_pattern, repl=' ', string=text)
	return without_urls

#ASCII
def accented_to_ascii(text):

	# apply unidecode function on text to convert
	# accented characters to ASCII values
	text = unidecode.unidecode(text)
	return text

### Contraction
def expand_contractions(text):
  ### We can use the dict : https://stackoverflow.com/questions/60901735/importerror-cannot-import-name-contraction-map-from-contractions
	return contractions.fix(text)


#######################################################################################################################################

### Suppresion de la ponctuation
import string

def remove_punctuation(text) :
  return text.translate(str.maketrans('', '', string.punctuation))

def preprocessing(text) :
  text = lower_case_convertion(text)
  text = remove_html_tags(text)
  text = remove_urls(text)
  text = accented_to_ascii(text)
  text =  expand_contractions(text)
  text = remove_punctuation(text)
  return text

def tag_words(sentence_tokenized):

    keywordSet = {"never", "nothing", "nowhere", "none", "not"}

    for word in sentence_tokenized:
      if (word in keywordSet) and (sentence_tokenized.index(word) < len(sentence_tokenized)-1):
        sentence_tokenized[sentence_tokenized.index(word)+1]=sentence_tokenized[sentence_tokenized.index(word)+1] + '_NEG'
        sentence_tokenized.pop(sentence_tokenized.index(word))
    return sentence_tokenized


def remove_stop_word(text) :
  stop_words = stopwords.words('english')
  tokens = word_tokenize(text)
  tokens_without_sw = [word for word in tokens if not word in stop_words]
  text = ' '.join(tokens)





def merge(dict1, dict2):
    for i in dict2.keys():
        dict1[i] = dict2[i]
    return dict1


def pos_tagging(text):
    ### POS_TAGGING
    lmtzr = WordNetLemmatizer()
    all_tag_words = {}
    tokens = word_tokenize(text)
    tagged = dict(nltk.pos_tag(tokens))
    all_tag_words = merge(tagged, all_tag_words)
    return all_tag_words


def get_wordnet_pos(treebank_tag):
    if treebank_tag.startswith('J'):
        return wordnet.ADJ
    elif treebank_tag.startswith('V'):
        return wordnet.VERB
    elif treebank_tag.startswith('N'):
        return wordnet.NOUN
    elif treebank_tag.startswith('R'):
        return wordnet.ADV
    else:
        return ''


# initialize lemmatizer object
lemma = WordNetLemmatizer()


### Lemmatize with POS tagging
def lemmatize(text, all_tag_words):
    tokens = ''.join(text).split()
    for i, token in enumerate(tokens):
        try:
            tokens[i] = lemma.lemmatize(token, get_wordnet_pos(all_tag_words[token]))
        except:
            pass
    text = ' '.join(tokens)
    return text



def inputs():

    texte_analyseur = st.sidebar.radio(
        "Quel texte analyser ? ",
        ('Avis dataset', 'Texte libre'))

    if texte_analyseur == 'Avis dataset':
        st.sidebar.write('Vous avez sélectionné Avis dataset.')
        dataset=random.choices(corpus,k=1)
        string=' '.join([str(item) for item in dataset])
        text = st.sidebar.text_area(label="Prediction des avis dataset 👇",value=string)
        dataset_randomly = random.choices(corpus, k=2)
        st.sidebar.info(dataset_randomly)
    if texte_analyseur == 'Texte libre':
        st.sidebar.write("Vous avez sélectionné Texte libre.")
        text = st.sidebar.text_area(label="Prediction de nouveaux avis 👇", value="Text to analyse..")
    features_nb = st.number_input(label="Nombre de topics", min_value=1, max_value=15)
    #features_nb = st.slider(label="Nombre de topics", min_value=1, max_value=15)
    button = st.button('Detecter le sujet d\'insatisfaction!')
    return features_nb,text,button


def predict(text: str, nb_features: int, NMF=pkl1, Vectorizer=pkl2):
    blob = TextBlob(text)
    polarity = blob.sentiment.polarity
    print(polarity)
    if polarity > 0:
         print("POLARITE:", polarity, " (AVIS POSITIF)")
         return None, polarity

    if polarity < 0:
        print("POLARITE:", polarity, " (AVIS NEGATIF)")
        avis = "AVIS NEGATIF"
        print(avis," and Polarity :",polarity)
        text = preprocessing(text)
        dict_pos = pos_tagging(text)
        text = lemmatize(text, dict_pos)
        text = [text]
        vect = Vectorizer.transform(text)
        nmf_features = list(list(NMF.transform(vect))[0])
        nmf_features_copy = nmf_features
        indexes = []

        for i in range(nb_features):
            index = nmf_features.index(max(nmf_features_copy))
            indexes.append(index)
            nmf_features_copy.pop(index)
            topics=[]
            topics_polarity=[]
        for i in indexes:
            data = labels_dict[i]
            #print(labels_dict[i])
            topics.append(data)
        topics_polarity.append(topics)
        topics_polarity.append(polarity)
        return topics_polarity




# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    st.header("Topic Modeling Application ")
    image = Image.open('./topic_modeling.png')
    st.image(image, caption='Topic Modeling')
    Positive_and_negative_img = Image.open('./positif-et-negatif.png')
    st.sidebar.image(Positive_and_negative_img,width=80)
    st.sidebar.header("Analyseur des avis")
    features_nb, text, button = inputs()
    if button:
        all_result= predict(text,features_nb)
        if all_result[1] > 0:
            st.info("\N{winking face} " + " POLARITE : " + f'{all_result[1]}' + "  (AVIS POSITIF)")
            st.info("TOPICS : " + f'{all_result[0]}', icon="ℹ")
        if all_result[1] < 0:
            st.error("POLARITE : " + f'{all_result[1]}' + "  (AVIS NEGATIF)", icon="😰")
            st.error("TOPICS : " + f'{all_result[0]}', icon="🚨")





    #predict("very bad nuggets chicken, too disapointed after many times came here", 3)



    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
