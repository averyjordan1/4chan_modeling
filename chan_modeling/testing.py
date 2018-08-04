from sklearn.datasets import fetch_20newsgroups
import ssl
import pprint
import gensim
import pprint
from archiving import MyThreads

# try:
#     _create_unverified_https_context = ssl._create_unverified_context
# except AttributeError:
#     pass
# else:
#     ssl._create_default_https_context = _create_unverified_https_context


# Lda = gensim.models.ldamodel.LdaModel
# model = Lda.load('ldamodel')
#
# pprint.pprint(model.top_topics())

docs = MyThreads('/Users/averyjordan/PycharmProjects/4chan_scrapping/biz_archive')

print(docs)