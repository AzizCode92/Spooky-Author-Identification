import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from time import time


train = pd.read_csv("../data/train.csv")
# Storing the first text element as a string
first_text = train.text.values[0]
first_text_list = nltk.word_tokenize(first_text)

#### stop words #######################
stopwords = nltk.corpus.stopwords.words('english')
first_text_list_cleaned = [word for word in first_text_list if word.lower() not in stopwords]
"""
print(first_text_list_cleaned)
print("="*90)
print("Length of original list: {0} words\n"
      "Length of list after stopwords removal: {1} words"
      .format(len(first_text_list), len(first_text_list_cleaned)))
"""
# https://www.kaggle.com/arthurtok/spooky-nlp-and-topic-modelling-tutorial

# inherited and subclassed the original Sklearn's CountVectorizer class and overwritten the build_analyzer
# by implementing the lemmatizer for each list in the raw text matrix.

lemm = WordNetLemmatizer()
class LemmaCountVectorizer(CountVectorizer):
    def build_analyzer(self):
        analyzer = super(LemmaCountVectorizer, self).build_analyzer()
        return lambda doc: (lemm.lemmatize(w) for w in analyzer(doc))

# Storing the entire training text in a list
text = list(train.text.values)

# Calling our overwritten Count vectorizer

tf_vectorizer = LemmaCountVectorizer(max_df=0.95,
                                     min_df=2,
                                     stop_words='english',
                                     decode_error='ignore')

# Learn the vocabulary dictionary and return term-document matrix.
tf = tf_vectorizer.fit_transform(text)


lda = LatentDirichletAllocation(n_components=11, max_iter=5,
                                learning_method = 'online',
                                learning_offset = 50.,
                                random_state = 0)

lda.fit(tf)




def print_top_words(model, feature_names, n_top_words):
    for topic_idx, topic in enumerate(model.components_):
        message = "Topic #%d: " % topic_idx
        message += " ".join([feature_names[i]
                             for i in topic.argsort()[:-n_top_words - 1:-1]])
        print(message)
    print()

t0 = time()
lda.fit(tf)
n_top_words = 20
print("done in %0.3fs." % (time() - t0))
print("\nTopics in LDA model:")
tf_feature_names = tf_vectorizer.get_feature_names()
print_top_words(lda, tf_feature_names, n_top_words)
