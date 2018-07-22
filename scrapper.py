import requests
import re

import time
from bs4 import BeautifulSoup
from Thread import Thread, Post
import graphviz as gv
from functional_helpers import *
from textwrap import wrap
from sentiment import analyze_post_sentiment, analyze_post_entities


# coins = [coin['name'].lower() for coin in requests.get('https://api.coinmarketcap.com/v1/ticker/').json()]
# print(coins)

"""
Helper functions 
"""
def clean(match):
    return re.sub('[><]+[0-9]+[><]*', '', BeautifulSoup(match, "html.parser").text)


def find_res(match):
    soup = BeautifulSoup(match, "html.parser")
    responses = soup.findAll('a')
    res_nums = set([])
    for response in responses:
        res_nums.add(response['href'][2:])
    return res_nums


def make_post(x):
    res = find_res(x.get('com', ""))
    return Post(title=str(x['no']),
                timestamp=x['time'],
                subject=clean(x.get('sub', "")),
                content=clean(x.get('com', "")),
                resto=res if res else {str(x.get('resto'))},
                responses=set([]))



# Gets a specific thread and then prints each of it's responses
def parse_thread(thread_num, visualize):
    """
    Given a thread_num of a thread on /biz/ on 4chan, this function creates a Thread representation of that thread.
    Useful for revealing sub thread conversations on a given thread. If visualize is enabled, creates a pdf file of the
    graph output, titled thread_thread_num.pdf.
    :param thread_num: the number of the thread to create
    :param visualize: boolean for visualization
    :return: the Thread object for future use
    """
    pg2 = requests.get('https://a.4cdn.org/biz/thread/{}.json'.format(thread_num)).json()


    # Maps all of the post JSON into threads
    posts = list(map(make_post, pg2['posts']))

    # Updates the posts to have responses attached to them
    list(map(lambda x: x.add_responses(list(filter(lambda y: x.title in y.resto, posts))), posts))

    # If the visualize argument is true, then render a graphviz digraph of the Thread
    if visualize:
        graph = gv.Digraph()
        list(map(lambda x: graph.node(name=str(x.title), label=str(x.content)), posts))
        for post in posts:
            for res in post.resto:
                graph.edge(str(post.title), str(res))
        graph.render('thread_{}'.format(thread_num))

        with open("thread_{}.json".format(thread_num), 'w+') as f:
            f.write(str(pg2))

    # Create the thread object, and mark the OP as the post that is a resto 0
    print('======================================================================')
    for post in posts:
        print(post)
    t = Thread(start_date=pg2.get('time'), title=pg2.get('title'),
           op=head(list(filter(lambda x: x.resto == {'0'} or x.resto == {0}, posts))))

    return t


# for post in parse_thread('4509898', False).op.responses:
#     print(analyze_post_sentiment(post.content))
    # analyze_post_entities(post.content)
#
# print(analyze_post_sentiment(parse_thread('4509898', False).op.content))


def simple_parse(thread_num, json_file=None):
    """
    Used for creating simple parse (just post content, no other fields) and returning their contents as a list.
    :param thread_num: the number of the thread to download content of.
    :return: a list of strings, where each string is the content of the post, and the list contains all the posts
            in the thread.
    """
    # Exponential back off to comply with the 4chan API usage policies.
    exp_backoff = 2
    if not json_file:
        pg2 = requests.get('https://a.4cdn.org/biz/thread/{}.json'.format(thread_num))
        while not pg2 and exp_backoff < 1024:
            exp_backoff = exp_backoff ** 2
            time.sleep(exp_backoff)

        posts = list(map(make_post, pg2.json()['posts']))
    else:
        pg2 = json_file
        posts = list(map(make_post, pg2['posts']))

    return posts
