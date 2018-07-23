import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from chan_modeling.archiving import MyThreads
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

Lda = gensim.models.ldamodel.LdaModel

TOPICS = 25
NUM_PASSES = 3
PRINT_N_WORDS = 10

def prepare_dictionary(threads):
    dictionary = corpora.Dictionary(threads)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in threads]
    return doc_term_matrix, dictionary
    
def train_lda_model(n_topics, dictionary, doc_term_matrix, n_passes, n_print_words):
    ldamodel = Lda(doc_term_matrix, num_topics=n_topics, id2word=dictionary, passes=n_passes)
    ldamodel.save('models/ldamodel')
    pprint.pprint(ldamodel.print_topics(num_topics=25, num_words=n_print_words))
    return ldamodel


def get_topics_for_document(doc, model, dictionary):
    bow = dictionary.doc2bow(doc)
    return model.get_document_topics(bow)