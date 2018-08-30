import os, sys
import numpy as np
import random
import math
import time

def readGraph(filename):
    start = time.time()
    print("entered readgraph index at {}".format(start))
    with open(filename, encoding='utf-8', mode='r') as fin:
        sims = {}
        info = fin.readline().split()
        oriInput = info[0]
        nAdjs = int(info[1].strip())

        for line in fin:
            info = line.split(':')
            word = info[0]
            if word not in sims:
                sims[word] = []
            for pair in info[1].split():
                adj, sim = pair.split(';')
                sims[word].append((adj, float(sim)))
        end = time.time()
        print("exited readgraph index after {}".format(end-start))
        return sims


# select pseudo-randomly a word from a list
# using the 'weight' value of the tuple to represent
# the item's likelihood to be chosen
def selectWord(words):
    max_value = sum([weight for vertex, weight in words])
    rand_num = random.uniform(0, max_value)
    pointer = 0
    for vertex, weight in words:
        pointer += weight
        if rand_num <= pointer:
            return vertex, weight


def expand(threads, outputName, graph, nTimes):
    start = time.time()
    print("entered expand at {}".format(start))
    outputFile = open(outputName, mode='w', encoding='utf-8')
    for doc in threads:
        candidates = {}
        for word in doc:
            for adj, sim in graph[word]:
                if adj not in candidates:
                    candidates[adj] = 0
                candidates[adj] += sim

        # selecting candidates that are not in the orginal doc
        candidates = [(w, candidates[w]) for w in candidates if w not in doc and candidates[w] > 0]

        # selecting new words
        newDoc = doc[:]
        while len(newDoc) < len(doc) * nTimes and len(candidates) > 0:
            pair = selectWord(candidates)
            word, sim = pair
            newDoc.append(word)
            candidates.remove(pair)  # without replacement

        # writing final pseudo document
        for w in newDoc:
            outputFile.write(w + ' ')
        outputFile.write('\n')
    outputFile.close()
    end = time.time()
    print("exited expand after {}".format(end-start))


def run_expand(threads, outputFile, graphName, nTimes):
    graph = readGraph(graphName)
    expand(threads, outputFile, graph, nTimes)
