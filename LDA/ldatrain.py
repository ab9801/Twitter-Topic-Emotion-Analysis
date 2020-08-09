import numpy as np
from joblib import dump, load
import pandas as pd
import re, nltk, spacy, gensim
import pickle
from sklearn.decomposition import LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import GridSearchCV
from pprint import pprint

import pyLDAvis
import pyLDAvis.sklearn
import matplotlib.pyplot as plt

df = pd.read_csv("textemotion.csv")


data = df.content.values.tolist()

data = [re.sub('\S*@\S*\s?', '', sent) for sent in data]

data = [re.sub('\s+', ' ', sent) for sent in data]

data = [re.sub("\'", "", sent) for sent in data]


def sent_to_words(sentences):
    for sentence in sentences:
        yield(gensim.utils.simple_preprocess(str(sentence), deacc=True))

data_words = list(sent_to_words(data))


def lemmatization(texts, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV']):
    texts_out = []
    for sent in texts:
        doc = nlp(" ".join(sent)) 
        texts_out.append(" ".join([token.lemma_ if token.lemma_ not in ['-PRON-'] else '' for token in doc if token.pos_ in allowed_postags]))
    return texts_out

nlp = spacy.load('en', disable=['parser', 'ner'])

data_lemmatized = lemmatization(data_words, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

#print(data_lemmatized[:2])

vectorizer = CountVectorizer(analyzer='word',       
                             min_df=10,
                             stop_words='english',
                             lowercase=True,
                             token_pattern='[a-zA-Z0-9]{3,}',
                            )

data_vectorized = vectorizer.fit_transform(data_lemmatized)
pickle.dump(data_vectorized, open("vdata.pickle", "wb"))
data_dense = data_vectorized.todense()

lda_model = LatentDirichletAllocation(n_topics=30,
                                      max_iter=10,               
                                      learning_method='online',
                                      learning_decay=0.9,
                                      random_state=100,
                                      batch_size=128,        
                                      evaluate_every = -1,   
                                      n_jobs = -1,   
                                     )
lda_output = lda_model.fit_transform(data_vectorized)

print(lda_model)


print("Log Likelihood: ", lda_model.score(data_vectorized))

print("Perplexity: ", lda_model.perplexity(data_vectorized))

pprint(lda_model.get_params())

dump(lda_model, 'ldamodel.joblib') 
#search_params = {'n_components': [10, 15, 20, 25, 30], 'learning_decay': [.5, .7, .9]}
'''
lda = LatentDirichletAllocation()

model = GridSearchCV(lda, param_grid=search_params)

model.fit(data_vectorized)

for d in data:
  docwords = ["Word" + str(i) for i in range(len(d))]
  dominant_topic = np.argmax(docwords.values, axis=1)

df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)
GridSearchCV(cv=None, error_score='raise',
       estimator=LatentDirichletAllocation(batch_size=128, doc_topic_prior=None,
             evaluate_every=-1, learning_decay=0.7, learning_method=None,
             learning_offset=10.0, max_doc_update_iter=100, max_iter=10,
             mean_change_tol=0.001, n_components=10, n_jobs=1,
             n_topics=None, perp_tol=0.1, random_state=None,
             topic_word_prior=None, total_samples=1000000.0, verbose=0),
       fit_params=None, iid=True, n_jobs=1,
       param_grid={'n_topics': [10, 15, 20, 25, 30], 'learning_decay': [0.5, 0.7, 0.9]},
       pre_dispatch='2*n_jobs', refit=True, return_train_score='warn',
       scoring=None, verbose=0)
search_params.update()
dump(model, 'gridsearchmodeltwo.joblib') 

data_vectorized = pickle.load(open("vdata.pickle", "rb"))
model = load('gridsearchmodel.joblib') 
ldamodel = model.best_estimator_

print("Best Model's Params: ", model.best_params_)

print("Best Log Likelihood Score: ", model.best_score_)

print("Model Perplexity: ", ldamodel.perplexity(data_vectorized))



#dump(model, 'gridsearchmodel.joblib')

n_topics = np.array([10, 15, 20, 25, 30])
log_likelyhoods = model.cv_results_['mean_test_score'].astype(int)
log_likelyhoods_5 = log_likelyhoods[:5]
log_likelyhoods_7 = log_likelyhoods[5:10]
log_likelyhoods_9 = log_likelyhoods[10:15]
# Show graph
plt.figure(figsize=(12, 8))
plt.plot(n_topics, log_likelyhoods_5, label='0.5')
plt.plot(n_topics, log_likelyhoods_7, label='0.7')
plt.plot(n_topics, log_likelyhoods_9, label='0.9')
plt.title("Choosing Optimal LDA Model")
plt.xlabel("Num Topics")
plt.ylabel("Log Likelyhood Scores")
plt.legend(title='Learning decay', loc='best')
plt.show()
'''
# Create Document - Topic Matrix
ldamodel = load('ldamodel.joblib') 
lda_output = ldamodel.transform(data_vectorized)
# column names
topicnames = ["Topic" + str(i) for i in range(ldamodel.n_topics)]

# index names
docnames = ["Doc" + str(i) for i in range(len(data))]

# Make the pandas dataframe
df_document_topic = pd.DataFrame(np.round(lda_output, 2), columns=topicnames, index=docnames)

# Get dominant topic for each document
dominant_topic = np.argmax(df_document_topic.values, axis=1)
df_document_topic['dominant_topic'] = dominant_topic

def color_green(val):
    color = 'green' if val > .1 else 'black'
    return 'color: {col}'.format(col=color)

def make_bold(val):
    weight = 700 if val > .1 else 400
    return 'font-weight: {weight}'.format(weight=weight)
df_document_topics = df_document_topic.head(15).style.applymap(color_green).applymap(make_bold)
print(df_document_topics)
df_topic_distribution = df_document_topic['dominant_topic'].value_counts().reset_index(name="Num Documents")
df_topic_distribution.columns = ['Topic Num', 'Num Documents']
print(df_topic_distribution)
visualisation = pyLDAvis.sklearn.prepare(ldamodel, data_vectorized, vectorizer, mds='tsne')
pyLDAvis.save_html(visualisation, 'ldv.html')

# Topic-Keyword Matrix
df_topic_keywords = pd.DataFrame(ldamodel.components_)

# Assign Column and Index
df_topic_keywords.columns = vectorizer.get_feature_names()
df_topic_keywords.index = topicnames
print(df_topic_keywords.head())


# Show top n keywords for each topic
def show_topics(vectorizer=vectorizer, lda_model=lda_model, n_words=20):
    keywords = np.array(vectorizer.get_feature_names())
    topic_keywords = []
    for topic_weights in lda_model.components_:
        top_keyword_locs = (-topic_weights).argsort()[:n_words]
        topic_keywords.append(keywords.take(top_keyword_locs))
    return topic_keywords

topic_keywords = show_topics(vectorizer=vectorizer, lda_model=ldamodel, n_words=15)        

# Topic - Keywords Dataframe
df_topic_keywords = pd.DataFrame(topic_keywords)
df_topic_keywords.columns = ['Word '+str(i) for i in range(df_topic_keywords.shape[1])]
df_topic_keywords.index = ['Topic '+str(i) for i in range(df_topic_keywords.shape[0])]
print(df_topic_keywords)








'''

# Define function to predict topic for a given text document.
nlp = spacy.load('en', disable=['parser', 'ner'])

def predict_topic(text, nlp=nlp):
    global sent_to_words
    global lemmatization

    # Step 1: Clean with simple_preprocess
    mytext_2 = list(sent_to_words(text))

    # Step 2: Lemmatize
    mytext_3 = lemmatization(mytext_2, allowed_postags=['NOUN', 'ADJ', 'VERB', 'ADV'])

    # Step 3: Vectorize transform
    mytext_4 = vectorizer.transform(mytext_3)

    # Step 4: LDA Transform
    topic_probability_scores = ldamodel.transform(mytext_4)
    topic = df_topic_keywords.iloc[np.argmax(topic_probability_scores), :].values.tolist()
    return topic, topic_probability_scores

# Predict the topic
mytext = ["Some text about christianity and bible"]
topic, prob_scores = predict_topic(text = mytext)
print(topic)
'''