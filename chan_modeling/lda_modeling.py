import gensim
from gensim import corpora
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
import string
from chan_modeling.archiving import MyThreads
import nltk
import pprint
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


Lda = gensim.models.ldamodel.LdaModel
CHUNK_SIZE=5000
NUM_PRINT_TOPICS=25


def prepare_dictionary(threads):
    dictionary = corpora.Dictionary(threads)
    doc_term_matrix = [dictionary.doc2bow(doc) for doc in threads]
    return doc_term_matrix, dictionary
    
    
def train_lda_model(n_topics, dictionary, doc_term_matrix, n_passes, n_print_words):
    ldamodel = Lda(doc_term_matrix, num_topics=n_topics, id2word=dictionary, passes=n_passes, chunksize=CHUNK_SIZE)
    ldamodel.save('models/ldamodel')
    pprint.pprint(ldamodel.print_topics(num_topics=NUM_PRINT_TOPICS, num_words=n_print_words))
    return ldamodel


def get_topics_for_document(doc, model, dictionary):
    bow = dictionary.doc2bow(doc)
    return model.get_document_topics(bow)


def prepare_dictionary_from_specific_files(all_files_directory, specific_files_text_file):
    f_names = None
    with open(specific_files_text_file) as f:
        f_names = f.readlines()
    cleaned_f_names = list(map(str.rstrip, f_names))    
    threads = MyThreads(all_files_directory, cleaned_f_names)
    dtm, dictionary = prepare_dictionary(threads)
    return dtm, dictionary