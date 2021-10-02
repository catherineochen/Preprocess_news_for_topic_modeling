# Preprocess_news_for_topic_modeling

The code provides common steps to organize and preprocess raw newspaper text files for topic modeling. The example is based on Ohio newspaper stories about 
hydraulic fracturing. The steps included are summarized below:

Step 1: Clean titles and content of news stories. 
- Clean lines containing copy right, dates, news agency, and other unhelpful information. 
- Standardize blank spaces between words

Step 2: Organize titles, news content and publication dates, construct a dataframe.

Step 3: Preprocess the news content
- Add lower case processing to the pipeline.
- Split documents into sentences.
- Lemmatize news texts (Bag Of Words approach).
- Add the uninformative letters/words to the stopword list, clean stop words.

Step 4: Generate bigrams of news content

Step 5: Run preliminary topic models to screen out common bigrams that are uninformative

