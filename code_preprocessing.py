
import glob
import re
import numpy as np
import csv
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

import pickle


import gensim
from gensim.models import LdaModel, LdaMulticore, HdpModel,ldaseqmodel
from gensim import corpora
from gensim.matutils import hellinger
from gensim.models import Phrases

import spacy

import nltk


#Import all the news stories (Ohio as example)

import zipfile
with zipfile.ZipFile('OH_all.zip', 'r') as zip_ref:
        zip_ref.extractall()   

news_list = glob.glob('*.txt')


##Step 1: Clean titile and content

#clean lines contaning news title & copy right. Other news papers may contain other irrelevant lines--check to organize the terms.

def should_remove_line(line, nonsense):
    return any([word in line for word in nonsense])

nonsense = ["Dateline", "Dispatch reporter", 'Copyright', 'Plain Dealer']


#create a list of news files by reading stories line by linw
blist = []

for file in news_list:
    ifile = open(file, 'r', encoding="utf8", errors='ignore').readlines()
    clean_ifile = []
    for line in ifile:
        if not should_remove_line(line, nonsense):
            clean_ifile.append(line)
    blist.append(clean_ifile)

len(blist)

#get rid of multiple blank spaces

dlist = []

for lists in blist:
    listss = ' '.join(str(e) for e in lists)
    dlist.append(listss)

alist = []

for m in dlist:
    n = " ".join(m.split())
    alist.append(n)

#create a list of news titles, clean the titles

title_lst = []

for title in news_list:
    split = re.split(r'\W+', title)
    lower = [word.lower() for word in split]
    join = '_'.join(lower)
    title_lst.append(join)

#merge the names with the content and turn it into a list

combined = list(zip(title_lst, alist))

#turn combined list into a pd data frame, add column names 

df_combined = pd.DataFrame(combined)
df_combined.columns = ['news', 'content']

#get rid of the .txt in news titles

df_combined['news'] = df_combined['news'].str.replace(r'_txt$', '')
print(df_combined.head())

# check whether the content looks good

df_combined.to_csv('_news_content.csv')


##Step 2: organize titile, news content and dates and construct a dataframe

#read in time label. Here, news_dates.csv consists of two cloulumns, news story titles and the dates of publication

opened_csv_file = open('news_dates.OH.csv', 'r', encoding="utf8", errors='ignore')
reader = csv.reader(opened_csv_file)

time_list = []
for row in reader:
        time_list.append(row)

#turn the list into a pandas dataframe

df_time = pd.DataFrame(time_list[1:], columns = time_list[0])
df_time.columns = ['news', 'date']
print(df_time.head())

#clean title in df_time: lower case all words

clean_title = df_time["news"].tolist()

new_title = []
for title in clean_title:
    split = re.split(r'\W+', title)
    lower = [word.lower() for word in split]
    join = '_'.join(lower)
    new_title.append(join)

df_time['news'] = new_title
df_time['news'] = df_time['news'].str.replace(r'_txt$', '')  

#if want to check whether things look good

df_time.to_csv('_news_time.csv')

#combined time label and content

df_merge = df_combined.merge(df_time, on='news', how = 'left')
df_merge.head()
len(df_merge)

#convert dates in the dataframe to standard time format

df_merge['date'] =  pd.to_datetime(df_merge['date'], 
                                              infer_datetime_format=True)

#check whether conversion worked

df_merge["date"].dt.year

#pickle the dataframe 
import pickle
with open("_oh_news.pickle", 'wb') as handle:
    pickle.dump(df_merge, handle, protocol=pickle.HIGHEST_PROTOCOL)
 
##Step 3: preprocess the news content

#preprocess the news texts

from spacy.lang.en.stop_words import STOP_WORDS
nlp = spacy.load("en_core_web_sm")

from nltk.corpus import stopwords
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize
ps = PorterStemmer()

# Add lower case processing to the pipeline

def lower_case_lemmas(doc) :
    for token in doc :
        token.lemma_ = token.lemma_.lower()
    return doc

nlp.add_pipe(lower_case_lemmas, name="lower_case_lemmas", after="tagger")


#define a function to break documents into sentences

alphabets= "([A-Za-z])"
prefixes = "(Mr|St|Mrs|Ms|Dr)[.]"
suffixes = "(Inc|Ltd|Jr|Sr|Co)"
starters = "(Mr|Mrs|Ms|Dr|He\s|She\s|It\s|They\s|Their\s|Our\s|We\s|But\s|However\s|That\s|This\s|Wherever)"
acronyms = "([A-Z][.][A-Z][.](?:[A-Z][.])?)"
websites = "[.](com|net|org|io|gov)"
digits = "([0-9])"

def split_into_sentences(text):
    text = " " + text + "  "
    text = text.replace("\n"," ")
    text = re.sub(prefixes,"\\1<prd>",text)
    text = re.sub(websites,"<prd>\\1",text)
    if "Ph.D" in text: text = text.replace("Ph.D.","Ph<prd>D<prd>")
    text = re.sub("\s" + alphabets + "[.] "," \\1<prd> ",text)
    text = re.sub(acronyms+" "+starters,"\\1<stop> \\2",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>\\3<prd>",text)
    text = re.sub(alphabets + "[.]" + alphabets + "[.]","\\1<prd>\\2<prd>",text)
    text = re.sub(" "+suffixes+"[.] "+starters," \\1<stop> \\2",text)
    text = re.sub(" "+suffixes+"[.]"," \\1<prd>",text)
    text = re.sub(" " + alphabets + "[.]"," \\1<prd>",text)
    text = re.sub(digits + "[.]" + digits,"\\1<prd>\\2",text)
    if "”" in text: text = text.replace(".”","”.")
    if "\"" in text: text = text.replace(".\"","\".")
    if "!" in text: text = text.replace("!\"","\"!")
    if "?" in text: text = text.replace("?\"","\"?")
    text = text.replace(".",".<stop>")
    text = text.replace("?","?<stop>")
    text = text.replace("!","!<stop>")
    text = text.replace("<prd>",".")
    sentences = text.split("<stop>")
    sentences = sentences[:-1]
    sentences = [s.strip() for s in sentences]
    return sentences

#split all news into sentences

df_merge['content'] = df_merge['content'].apply(split_into_sentences)

#get a list of all articles 

all_articles = []
for article in df_merge['content']:
    all_articles.append(article)

# create a list of lemmatized news (Bag Of Words approach), spacy functions 

all_news = []

for row in df_merge['content']:
    news = []
    for sentence in row:
        text = []
        doc = nlp(sentence)
        for word in doc:
            if not word.is_stop and not word.is_punct and not word.like_num:
                text.append(word.lemma_)
        news.append(text)
    all_news.append(news)       

stoplist = nltk.corpus.stopwords.words('english')

#add the uninformative letters/words to the stopword list

new_Stoplist = ['a', 'b', 'c', 'd', 'e','f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p', 'q', 'r', 's', 
                't', 'u', 'v', 'w', 'x', 'y', 'z', 'sen.', 'rep.', 'gov.', 'dem.', 'say',
               'ohio', 'year', '$']

for i in new_Stoplist:
    stoplist.append(i)
print(stoplist)

#delete stop words from all_news
all_news_cleaned = []

for row in all_news:
    news = []
    for sentence in row:
        text = []
        for word in sentence:
            if word not in stoplist:
                text.append(word)
        news.append(text)
    all_news_cleaned.append(news)

print(all_news_cleaned[3])

#join the cleaned and lemmatized sentences in each article

def join_lem(sentence):
    return ' '.join(sentence)

#rejoin all sentences in the lemmatized news
join_all_news = []

for news in all_news_cleaned:
    join_news = []
    for sentence in news:
        join_sent = join_lem(sentence)
        join_news.append(join_sent)
    join_all_news.append(join_news)

join_all_news[3]

##Step 4: generate bigrams of news content 

from nltk import word_tokenize 
from nltk.util import ngrams

#generate bigrams in each news article

news_bigram_raw = []

for news in join_all_news:
    new_big = [b for l in news for b in zip(l.split(" ")[:-1], l.split(" ")[1:])]
    biggie_ = ['_'.join(y) for y in new_big]
    news_bigram_raw.append(biggie_)

#Single out uninformative bigrams

uninfo = ['oil_gas', 'natural_gas', 'hydraulic_fracturing']

news_bigram = []

for news in news_bigram_raw:
    info = []
    for bigram in news:
        if bigram not in uninfo:
            info.append(bigram)
    news_bigram.append(info)

df_merge['content'] = news_bigram

##Step 5: run preliminary topic models to screen out uninformative and common bigrams

#run an hdp model and pick out common and uninformtive bigrams such as localities to be screened out
#repeat after df_merge['content']=news_info

#create indexed list of content, pickle then read as bytes

BOW_content_bigram = []
for i in range(len(df_merge['content'])):
    doc = df_merge['content'].iloc[i]
    BOW_content_bigram.append(doc)

with open("_BOW_bigram_trial.pickle", 'wb') as handle:
    pickle.dump(BOW_content_bigram, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open("_BOW_bigram_trial.pickle", 'rb') as handle:
    BOW_bytes = pickle.load(handle)

#Creat dictionary for Bag of Words

lda_dictionary = corpora.Dictionary(BOW_bytes)
corpus = [lda_dictionary.doc2bow(text) for text in BOW_bytes]

#try hdp model

lda_hdp = HdpModel(corpus=corpus, id2word=lda_dictionary, chunksize = 100, random_state = 107)
lda_hdp.optimal_ordering()
lda_hdp.print_topics(num_topics = 15, num_words = 10)

#get rid of uninformative bigrams

uninform = ['monroe_county', 'dominion_east', 'mahoning_county', 'new_york', 'john_kasich',
           'harrison_county']

news_info = []

for news in news_bigram:
    info = []
    for bigram in news:
        if bigram not in uninform:
            info.append(bigram)
    news_info.append(info)
    
df_merge['content'] = news_info


#additional code to make sure uninformative bigrams are indeed removed

df_merge = pd.read_pickle(r'_oh_bow_bigram.pickle')
news_bigram = df_merge['content']
uninform = ['gas_drilling', 'shale_gas']

news_info = []
for news in news_bigram:
    info = []
    for bigram in news:
        if bigram not in uninform:
            info.append(bigram)
    news_info.append(info)

df_merge['content'] = news_info

with open("_oh_bow_bigram.pickle", 'wb') as handle:
    pickle.dump(df_merge, handle, protocol=pickle.HIGHEST_PROTOCOL)
