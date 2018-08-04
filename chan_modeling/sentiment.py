from google.cloud import language
from google.cloud.language import enums
from google.cloud.language import types
import os

os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = '/Users/averyjordan/PycharmProjects/4chan_scrapping/transcription-2320e1724fa7.json'

# Instantiates a client
client = language.LanguageServiceClient()

def analyze_post_sentiment(post_text):

    text = post_text
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the text
    sentiment = client.analyze_sentiment(document=document).document_sentiment

    return 'Text: {} \nSentiment: {}, {}\n'.format(text, sentiment.score, sentiment.magnitude)


def analyze_post_entities(post_text):
    text = post_text
    document = types.Document(
        content=text,
        type=enums.Document.Type.PLAIN_TEXT)

    # Detects the sentiment of the text
    entities = client.analyze_entities(document=document).entities

    for entity in entities:
        print('=' * 20)
        print('         name: {0}'.format(entity.name))
        print('         type: {0}'.format(entity.type))
        print('     metadata: {0}'.format(entity.metadata))
        print('     salience: {0}'.format(entity.salience))


# with open('biz_archive/thread_4240043_archive.txt', 'r+') as f:
#     print(analyze_post_entities(" ".join(f.readlines())))
