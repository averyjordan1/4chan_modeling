import os, sys
import numpy as np
import time

# inverted index for dataset
def createIndex(threads):
    start = time.time()
    print('entered create index at {}'.format(start))
    index = {}  # inverted index
    docID = 0
    for words in threads:
        for w in words:
            if w not in index:
                index[w] = set()
            index[w].add(docID)
        docID += 1
    end = time.time()
    print("excited create index after {}".format(end-start))
    return index


# creates word similarity graph with Jaccard index
# graph is an adjacency list
# indexes in list comes from word indexes in sorted vocab
def createGraph(index, nAdjs):
    start = time.time()
    print('entered create graph at {}'.format(start))
    vocab = [*index]
    vocab.sort()
    graph = [[] for i in range(len(vocab))]
    print('sorted vocabulary and initialized graph after {}'.format(time.time() - start))
    for i1 in range(len(vocab) - 1):
        for i2 in range(i1 + 1, len(vocab)):
            w1 = vocab[i1]
            w2 = vocab[i2]

            docs1 = index[w1]
            docs2 = index[w2]

            intersec = len(docs1.intersection(docs2))
            union = len(docs1.union(docs2))
            jacc = intersec / float(union)

            if intersec > 0:
                graph[i1].append((jacc, i2))
                graph[i2].append((jacc, i1))
    print('sorted vocabulary and initialized graph after {}'.format(time.time() - start))
   
    # keeps only the nAdjs best adjs for each word
    for i1 in range(len(vocab) - 1):
        graph[i1].sort(reverse=True)
        if len(graph[i1]) > nAdjs:
            graph[i1] = graph[i1][:nAdjs]
    end = time.time()
    print("excited create graph after {}".format(end-start))

    return graph, vocab


# format:
# filename nAdjs
# word:adj;sim adj2;sim adj3;sim
def storeGraph(filename, name_output, graph, vocab, nAdjs):
    name_output += "_graph" + '.txt'
    with open(name_output, 'w') as fout:
        fout.write("%s %s\n" % (filename, nAdjs))
        for i in range(len(vocab)):
            w = vocab[i]
            fout.write(w + ':')
            for sim, adj in graph[i]:
                fout.write("%s;%f " % (vocab[adj], sim))
            fout.write('\n')
    return name_output


def run_create_graph(threads, nAdjs, outname, filename):
    index = createIndex(threads)
    graph, vocab = createGraph(index, nAdjs)
    return storeGraph(filename, outname, graph, vocab, nAdjs)
