import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from archiving import MyThreads
import logging
import nltk
import pprint
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
import time

#
# import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
# nltk.download('wordnet')

"""
LDA attempt 1
"""

before_docs = time.time()
docs = MyThreads('/Users/averyjordan/PycharmProjects/4chan_scrapping/biz_archive')
after_docs = time.time()
print("After docs has finished: {}".format(after_docs - before_docs))


before_dictionary = time.time()
dictionary = corpora.Dictionary(docs)
after_dictionary = time.time()
print("After dictionary has finished: {}".format(after_dictionary - before_dictionary))

before_term_matrix = time.time()
doc_term_matrix = [dictionary.doc2bow(doc) for doc in docs]
after_term_matrix = time.time()
print("After term matrix has finished: {}".format(after_term_matrix - before_term_matrix))


Lda = gensim.models.ldamodel.LdaModel

TOPICS = 25
NUM_PASSES = 3
PRINT_N_WORDS = 10



def train_lda_model(n_topics, dictionary, n_passes, n_print_words):
    ldamodel = Lda(doc_term_matrix, num_topics=n_topics, id2word=dictionary, passes=n_passes)
    ldamodel.save('/Users/averyjordan/PycharmProjects/4chan_scrapping/ldamodel')
    pprint.pprint(ldamodel.print_topics(num_topics=25, num_words=n_print_words))



for n_topics in range(1, 51, 5):
    train_lda_model(n_topics, dictionary, NUM_PASSES, PRINT_N_WORDS)
