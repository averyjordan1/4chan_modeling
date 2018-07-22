import string

import requests
# import nltk.data
# from scrapper import parse_thread, simple_parse
import os
import time
import gensim
import re
import logging
import csv
from markovipy import MarkoviPy
from copy import deepcopy

from nltk import WordNetLemmatizer
from nltk.corpus import stopwords

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)


# CODE for importing the SSL thing to fix a nltk error

# import ssl
# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context
# nltk.download('punkt')

def add_data_files():
    # Get all of the archived threads on 4chan/biz
    j = requests.get("https://a.4cdn.org/biz/archive.json").json()

    # Set the tokenizer
    tokenizer = nltk.data.load('tokenizers/punkt/english.pickle')

    # Get all the existing files in the archive
    fls = os.listdir('/Users/averyjordan/PycharmProjects/4chan_scrapping/biz_archive')

    print(fls)
    # Iterate over all the threads in the archive

    for thread_num in j:

        # Create output filename, and check if it already exists, if so, don't pull it again
        temp_name = 'thread_{}_archive.txt'.format(thread_num)
        if temp_name not in fls:

            # Pur each sentence on a new line
            data = "\n".join(list(map(lambda x: x.content, simple_parse(thread_num))))
            sentences = "\n".join(tokenizer.tokenize(data))

            # Write them to a file
            with open("biz_archive/{}".format(temp_name), 'w+') as f:
                f.write(sentences)

            # Wait to comply with 4chan guidelines
            time.sleep(1)
            print("Added: {}".format(thread_num))
        else:
            print("Skipping: {}".format(thread_num))


def clean_existing_files():
    sents = MySentences('/Users/averyjordan/PycharmProjects/4chan_scrapping/biz_archive')

    with open('cleaned_files/mega_file.txt', 'w+') as f:
        for sent in sents:
            # print(sent)
            new_sent = []
            for word in sent:
                punc = '<>",/()'
                new_word = word.strip(punc)
                new_word = new_word.replace('>', '')
                new_word = new_word.strip('0123456789')
                new_sent.append(new_word.lower())
            f.write(" ".join(new_sent) + '\n')



# Simple iterator class for using less intense memory usage
class MySentences(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if fname.endswith('.txt'):
                for line in open(os.path.join(self.dirname, fname), encoding='utf-8'):
                    yield line.split()


stop = set(stopwords.words('english'))
exclude = set(string.punctuation)
lemma = WordNetLemmatizer()


def clean(doc):
    doc = " ".join(doc)
    stop_free = " ".join([i for i in doc.lower().split() if i not in stop])
    punc_free = ''.join(ch for ch in stop_free if ch not in exclude)
    normalized = " ".join(lemma.lemmatize(word) for word in punc_free.split())
    return normalized.split()

class MyThreads(object):
    def __init__(self, dirname):
        self.dirname = dirname

    def __iter__(self):
        for fname in os.listdir(self.dirname):
            if fname.endswith('.txt'):
                lines = []
                for line in open(os.path.join(self.dirname, fname), encoding='utf-8'):
                    lines.append(line)
                yield clean(lines)




def train(s, c, fname=None):
    dir = '/Users/averyjordan/PycharmProjects/4chan_scrapping/cleaned_files'
    sentences = MySentences(dir)
    phrases = gensim.models.Phrases(sentences)
    bigrams = gensim.models.phrases.Phraser(phrases)
    model = gensim.models.Word2Vec(bigrams[sentences], size=s, workers=4, min_count=c, hs=1)
    if fname:
        model.save('/Users/averyjordan/PycharmProjects/4chan_scrapping/models/{}_model'.format(fname))
    else:
        model.save('/Users/averyjordan/PycharmProjects/4chan_scrapping/models/4chan_biz_size_{}_count_{}'.format(s, c))

    # print(model.corpus_count)
    vectors = model.wv
    if fname:
        vectors.save('/Users/averyjordan/PycharmProjects/4chan_scrapping/models/{}_vectors'.format(fname))
    else:
        vectors.save('/Users/averyjordan/PycharmProjects/4chan_scrapping/models/4chan_biz_vectors_size_{}_count_{}'.format(s, c))


def analyze(s, c):
    vectors = gensim.models.KeyedVectors.load(
        '/Users/averyjordan/PycharmProjects/4chan_scrapping/models/4chan_biz_vectors_size_{}_count_{}'.format(s, c))
    with open('models/tests/size_{}_count_{}.txt'.format(s, c), 'w+') as f:

        f.write("Test results for model of size {} and min_count {}\n".format(s, c))
        f.write("=================================================\n")
        f.write("Test: positive=['pajeet', 'scam'], negative=['chad']\n")
        for word in vectors.most_similar_cosmul(positive=['pajeet', 'scam'], negative=['chad'], topn=10):
            f.write(str(word) + "\n")

        f.write("=================================================\n")
        f.write("Test: positive=['bitcoin', 'btc'], negative=['altcoin']\n")
        for word in vectors.most_similar_cosmul(positive=['bitcoin', 'btc'], negative=['altcoin'], topn=10):
            f.write(str(word) + "\n")

        f.write("=================================================\n")
        f.write("Test: positive=['bitcoin', 'btc'], negative=['ether']\n")
        for word in vectors.most_similar_cosmul(positive=['bitcoin', 'btc'], negative=['ethereum'], topn=10):
            f.write(str(word) + "\n")

        sim_list = ['ripple', 'bitcoin', 'ethereum', 'pajeet', 'chad', 'altcoin']
        for sim in sim_list:
            f.write("=================================================\n")
            f.write("Test: similarity to '{}'\n".format(sim))
            for word in vectors.similar_by_word(sim, topn=10):
                f.write(str(word) + "\n")


def parse_result_file(file):
    lines = file.readlines()
    return {'chad_analogy': lines[3], 'altcoin_analogy': lines[15], 'ripple': lines[39], 'bitcoin': lines[51],
            'ethereum': lines[63], 'chad': lines[87], 'altcoin': lines[99]}


def evaluate_test_results():
    all_results = []
    for fname in os.listdir('/Users/averyjordan/PycharmProjects/4chan_scrapping/models/tests'):
        if fname.endswith('.txt'):
            with open('models/tests/{}'.format(fname), 'r') as f:
                results = parse_result_file(f)
                results['file'] = fname
                all_results.append(results)

    with open('test_results.csv', 'w+') as csvfile:
        writer = csv.DictWriter(csvfile, all_results[0].keys())
        writer.writeheader()
        for result in all_results:
            writer.writerow(result)


# evaluate_test_results()

def find_similarities(model_nums, word):
    vectors = gensim.models.KeyedVectors.load(
        '/Users/averyjordan/PycharmProjects/4chan_scrapping/models/4chan_biz_vectors_size_{}_count_{}'.format(
            model_nums[0], model_nums[1]))
    for vec in vectors.similar_by_word(word, topn=20):
        print(vec)


def find_analogy(model_nums, positives, negatives):
    vectors = gensim.models.KeyedVectors.load(
        '/Users/averyjordan/PycharmProjects/4chan_scrapping/models/4chan_biz_vectors_size_{}_count_{}'.format(
            model_nums[0], model_nums[1]))
    for word in vectors.most_similar_cosmul(positive=positives, negative=negatives, topn=20):
        print(word)


def test_accuracy(words_list):
    with open('word_list_test.csv', 'w+') as csvfile:
        # Word list
        wlist2 = deepcopy(words_list)
        wlist2.insert(0, 'file')

        # Establish csv writer
        writer = csv.DictWriter(csvfile, wlist2)
        writer.writeheader()
        for s in range(50, 301, 5):
            for c in range(5, 11, 5):
                # Get the requisite models vectors
                vectors = gensim.models.KeyedVectors.load(
                    '/Users/averyjordan/PycharmProjects/4chan_scrapping/models/4chan_biz_vectors_size_{}_count_{}'.format(
                        s, c))

                # Set up the row dictionary with file name
                row_dict = {'file': '{}_size_{}_count'.format(s,c)}

                # Process all the words in the word list
                for word in words_list:
                    vec = vectors.similar_by_word(word, 1)
                    row_dict[word] = vec[0][0]

                # Write the words to the CSV
                writer.writerow(row_dict)


def bigram_creation():
    phrases = gensim.models.Phrases(MySentences('/Users/averyjordan/PycharmProjects/4chan_scrapping/cleaned_files'))
    bigrams = gensim.models.phrases.Phraser(phrases)
    trigrams = gensim.models.phrases.Phraser(bigrams[MySentences('/Users/averyjordan/PycharmProjects/4chan_scrapping/cleaned_files')])
    sent = "pump and dump schemes are evil.".split()
    print(trigrams[sent])

# bigram_creation()

#
# with open('test_words', 'r+') as f:
#     words = [line.strip() for line in f.readlines()]
#
# test_accuracy(words)


# train(90,5, fname="bigram_test")

# print(find_similarities((150,5), 'goy'))