# Keyword and Topic Extraction from Movie Reviews

## Dataset
**Large Movie Review Dataset:**
This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. We provide a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided. <br>
<a href = 'https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz'> Click to download the dataset </a> <br>

## Tools and Libraries

<ul>
  <li>NLTK (Words tokenizer, stopwords and WordNetLemmatizer)</li>
  <li>Seaborn and Matplotlib</li>
  <li>Gensim libraries</li>
  <li>WordCloud</li>
  <li>Sentence Transformer and sklearn</li>
  <li>Numpy and Pandas</li>
  <li>PyLDAViz</li>
</ul>

## Steps involved in the project
### 1. Pre-processing
<ul>
  <li>Remove stop-words with NLTK</li>
  <li>Remove number from text with regular expression function</li>
  <li>Lower the text and remove words lower than 3 letters</li>
  <li>Bring the text back to it's base word via lemmatization</li>
</ul>

### 2. EDA
<ul>
  <li>Did some visualization,  wordcloud to uderstand the common words in the review</li>
  <li>Meta Analysis of the data</li>
  <li>Data Distribution</li>
</ul>

### 3. Topic Modelling using LDA
Created a dictionatry and a corpus with the review text. 
LDA model (LDAMulticore) was used for topic modelling and vizualization was done using PyLDAViz library.

![screencapture-file-C-Users-ADMIN-Desktop-Swetha-Academics-Keyword-Topic-Extraction-from-Movie-Reviews-Lda-viz-html-2021-06-11-18_43_39](https://user-images.githubusercontent.com/68152189/121691850-15fa9f80-cae5-11eb-85b9-a3a39f28e85d.png)


### 4. Keywords Extraction using Distilbert
Distilbert is used as it has shown great performance in similarity tasks, which is what we are aiming for with keywords extraction.
To find the candidates that are most similar to the document, cosine similarity is used. The most similar candidates to the document are good keywords for representing each review and the keywords are thus got from the candidates. On entering the review number corresponding keywords are returned.

<img width="568" alt="Capture" src="https://user-images.githubusercontent.com/68152189/121651275-0fa1fe80-cab8-11eb-929c-1ea0644c533a.PNG">
