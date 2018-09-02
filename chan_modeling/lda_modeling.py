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
    """
    Returns the topics associated with a given document from a given model.
    """
    bow = dictionary.doc2bow(doc)
    return model.get_document_topics(bow)


def prepare_dictionary_from_specific_files(all_files_directory, f_names):
    """
    Given a list of specific file names (e.g., that have been selected on some condition), create a 
    document term matrix, and a dictionary for training an LDA model. 
    """
    cleaned_f_names = list(map(str.rstrip, f_names))    
    threads = MyThreads(all_files_directory, cleaned_f_names)
    dtm, dictionary = prepare_dictionary(threads)
    return dtm, dictionary


def select_files_on_word_count(all_files_counts, predicate):
    """
    From a file containing the counts of each possible thread, selects the desired files based on a user 
    given predicate that filters on length. Predicate should return true given a length that is desired.
    """
    selected_files = []
    with open(all_files_counts) as f:
        for txt_file in f:
            length, file_name = txt_file.lstrip().split(' ')
            if predicate(int(length)):
                selected_files.append(file_name.strip())
    return selected_files
                
            