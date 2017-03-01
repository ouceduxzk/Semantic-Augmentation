import pickle, os
import re
import matplotlib.pyplot as plt
import pickle
import numpy as np

def getDict():
    title2ind, ind2title = {}, {}
    if not os._exists('title_map.pkl'):
        lines = open('title.txt', 'r').readlines()
        lines = [x.strip() for x in lines]
        ind2title = dict(enumerate(lines))
        title2ind = dict([(line, i) for i, line in enumerate(lines)])

        with open('title_map.pkl', 'w') as fp :
            pickle.dump(title2ind, fp)
            pickle.dump(ind2title,  fp)
    else:
        with open('title_map.pkl', 'r') as fp :
            pickle.load(title2ind, fp)
            pickle.load(ind2title, fp)
    return title2ind, ind2title


def get_concept_samples():
    lines = open('concept_samples.txt', 'r').readlines()
    return [ x.strip() for x in lines]

def get_query_result(pkl_fn):
    data = pickle.load(open(pkl_fn, 'rb'))
    concepts = [x[0] for x in data]
    sims =     [x[1] for x in data]
    #indices =  [x[2] for x in data]
    return concepts,  sims

# def cal_tag_tag_sim(tag_pkls):
#     Tag_tag_topk_sims = []
#     for tag_pkl in tag_pkls:
#         print(tag_pkl)
#         concepts, sims = get_query_result('query_pkl/' + tag_pkl)
#         gradient = np.gradient(sorted(sims))
#         sorted_sim_indices = np.argsort(sims)
#         topk_gradient_indices = np.where(gradient > 0.005)[0]
#         topk_sim_indices = sorted_sim_indices[topk_gradient_indices][:-1]
#         topk_sim = np.array(sims)[topk_sim_indices]
#         topk_concepts = np.array(concepts)[topk_sim_indices]
#         Tag_tag_topk_sims.append( [topk_sim_indices, topk_concepts, topk_sim])
#         #print(topk_sim_indices, topk_concepts, topk_sim)
#     pickle.dump(Tag_tag_topk_sims, open('Tag_tag_sim_topk.pkl', 'wb'))
#     return Tag_tag_topk_sims


def cal_tag_tag_sim(tag_pkls):
    Tag_tag_topk_sims = []
    for tag_pkl in tag_pkls:
        print(tag_pkl)
        concepts, sims = get_query_result('query_pkl/' + tag_pkl)

        sorted_sim_indices = np.argsort(sims)

        threhold = np.percentile(sims, 95)

        topk_sim = np.sort(sims[sims > threhold])[::-1]

        topk_quantile_indices = np.nonzero(np.in1d(sims, topk_sim ))[0]

        topk_sim_indices = np.sort(topk_quantile_indices)[:-1]

        topk_concepts = np.array(concepts)[topk_sim_indices]

        Tag_tag_topk_sims.append( [topk_sim_indices, topk_concepts, topk_sim])

    pickle.dump(Tag_tag_topk_sims, open('Tag_tag_sim_topk.pkl', 'wb'))
    return Tag_tag_topk_sims

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
    result = {}
    with open("wiki_tfidf/_bow.mm") as infile:
        for i, piece in enumerate(infile):

            tmp = piece.strip().split()

            if i < 2:
                continue
            word = tmp[1]
            if i % 10000 == 0 :
                print(i, word)
            if word in result.keys():
                result[word] += 1
            else:
                result[word] = 1
    pickle.dump(result, open('idf.pkl', 'wb'))
