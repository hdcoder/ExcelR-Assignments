import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import string
import spacy
import nltk

from nltk.tokenize import TweetTokenizer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.tokenize import sent_tokenize

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer

import re
import warnings

from wordcloud import WordCloud
from textblob import TextBlob

def get_cleanText(text):
    text = re.sub(r'@[A-Za-z0-9]+','',text)
    text = re.sub(r'#','',text)
    text = re.sub(r'RT[\s]+','',text)
    text = re.sub(r'https?:\/\/\S+','',text)
    text=  re.sub("[^A-Za-z" "]+"," ",text)
    return text

def  get_Subjectivity(text):
    return TextBlob(text).sentiment.subjectivity

def get_Polarity(text):
    return TextBlob(text).sentiment.polarity

def get_score(score):
    score = float(score)
    if (score < 0):
        return 'Negative'
    elif (score == 0):
        return 'Neutral'
    elif score > 0:
        return 'Positive'

Elon = pd.read_csv("Elon_musk.csv")
print(Elon)
print()

Elon.drop('Unnamed: 0', axis=1, inplace=True)
print(Elon)
print()

Elon = [text.strip() for text in Elon.Text]
Elon = [text for text in Elon if text]
print(Elon[0:10])
print()

texts = ''.join(Elon)

Tokenizer = TweetTokenizer(strip_handles=True)
Tokens = Tokenizer.tokenize(texts)
Tokens = str(Tokens)

for c in string.punctuation:
    Tokens=  Tokens.replace(c,"")

Wo_links = re.sub(r'http\s+', '', Tokens)
Wo_links = Wo_links.lower()

Wo_n = re.sub("[0-9" "]+"," ", Wo_links).lower()
Wo_s = re.sub("[^A-Za-z" "]+", " ", Wo_n).lower()

nltk.download('stopwords')
nltk.download('punkt')

FT = word_tokenize(Wo_s)

Without_sw = [ word for word in FT if word not in stopwords.words('english')]

NLP = spacy.load('en_core_web_sm')
LEM = NLP(' '.join(Without_sw))

Simple_words = [Token.lemma_ for Token in LEM]

print(Simple_words)
print()

TF = CountVectorizer()

Vectors = TF.fit_transform(Simple_words)

TD = TfidfVectorizer()

Vectors1 = TD.fit_transform(Simple_words)

warnings.filterwarnings("ignore")

Feature_Names = TF.get_feature_names()

Dense = Vectors1.todense()
Denselist = Dense.tolist()

DF1 = pd.DataFrame(Denselist, columns = Feature_Names)

print(DF1)
print()

W_list = ' '.join(DF1)

Wordcloud = WordCloud(background_color='black',width=2000,height=1600).generate(W_list)

plt.imshow(Wordcloud)

for i in LEM[0:10]:
    print(i, i.pos_)

print()

Nouns_verbs = [token.text for token in LEM if token.pos_ in ('NOUN', 'VERB')]

CV = CountVectorizer()
X = CV.fit_transform(Nouns_verbs)
Sum_words = X.sum(axis=0).tolist()[0]
Words_freq = [(Word, Sum_words[Idx]) for Word, Idx in CV.vocabulary_.items()]
Words_freq = sorted(Words_freq, key = lambda x: x[1], reverse=True)
WF_DF1 = pd.DataFrame(Words_freq)
WF_DF1.columns = ['word', 'count']

print(WF_DF1[0:10])
print()

WF_DF1[0:10].plot.bar(x='word', figsize=(12,8), title = 'Top verbs and nouns')

Elon = pd.DataFrame(Elon, columns = ['text'])

print(Elon)
print()

Elon['clean_text'] = Elon['text'].apply(get_cleanText)

print(Elon['clean_text'])
print()

Elon['Subjectivity'] = Elon['clean_text'].apply(get_Subjectivity)
Elon['Polarity'] = Elon['clean_text'].apply(get_Polarity)

print(Elon)
print()

Allwords = ' '.join([word for word in Elon['clean_text']])
WC = WordCloud(width=700,height=500,random_state=4,max_font_size=150).generate(Allwords)

plt.figure()
plt.imshow(WC,interpolation='bilinear')
plt.axis('off')
plt.show()

Elon['Sentiment'] = Elon['Polarity'].apply(get_score)

print(Elon)
print()

plt.figure()
plt.title("Sentimental Analysis")
plt.xlabel("Sentiment")
plt.ylabel("Count")
plt.style.use('seaborn-dark-palette')
Elon['Sentiment'].value_counts().plot(kind='bar');

Positive = Elon[Elon['Polarity']>0]
print(Positive['text'])
print()

Negatives = Elon[Elon['Polarity']<0]
print(Negatives['text'])
print()

PTweets = Positive['text']
Score = round(PTweets.shape[0]/Elon.shape[0] *100)
print("The percentage of positive tweets is {} %.".format(Score))
print()

NTweets = Negatives['text']
ScoreN = round(NTweets.shape[0]/Elon.shape[0] *100)
print("The percentage of negative tweets is {} %.".format(ScoreN))
print()

Neutral = Elon[Elon['Polarity']==0]
Neutral = Neutral['text']
NN = round(Neutral.shape[0]/Elon.shape[0] *100)
print("The percentage of Neutral tweets is {} %.".format(NN))