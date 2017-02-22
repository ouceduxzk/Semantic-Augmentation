import pickle, os
import re
import matplotlib.pyplot as plt
import pickle
import numpy as np

def plot_query_result(pkl_fn, outputfn):
    data = pickle.load(open(pkl_fn, 'rb'))
    concetps = [x[0] for x in data]
    sims =     [x[1] for x in data]
    plt.plot(sims)
    if not os.path.exists('query_sim'):
        os.mkdir('query_sim')

    #plt.savefig('query_sim/' + outputfn)
    ##plt.close()

def check_wierd_query(pkl_fn):
    data = pickle.load(open(pkl_fn, 'rb'))
    concetps = [x[0] for x in data]
    sims = [x[1] for x in data]
    if np.mean(sims) > 0.8:
        print(pkl_fn, data)

def generate_query_doc_vector(idx):
    os.system('grep "^' + str(idx) + '" > tmp.txt')
    lines = open('tmp.txt', 'rb').readlines()
    vec = [0] * 100000
    indices = []
    for line in lines :
        docid, word_id, tfidf = line.strip().split()
        vec[int(word_id)] = tfidf 
    return vec 

def extract_title():
    #data = wikicorpus.extract_pages('enwiki-latest-pages-articles.xml.bz2')
    data = open('title.txt', 'rb').readlines()
    title2ind = dict([ (x, i+1) for i, x in enumerate(data)])
    pickle.dump(title2ind, open('title2ind.pkl', 'wb'))

def calculate_idf_all():

    f = open('wiki_tfidf/_bow.mm')
    result = {}
    for piece in read_in_chunks(f):
        tmp = piece.strip().split()
        if re.search('[a-zA-Z]', tmp[1]):
            continue
        word = tmp[1]
        if word in result.keys():
            result[word] += 1
        else:
            result[word] = 1

    pickle.dump(result, open('idf.pkl', 'wb'))
