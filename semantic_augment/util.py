#!/usr/bin/env python
# -*- coding: utf-8 -*-
import pickle, os
import re
import matplotlib.pyplot as plt
import pickle
import numpy as np
import codecs

############# correct some mismatch of wiki concept due to the heterogeneity of those concepts#####################
correction = {}
correction['data classification']  = 'data classification (data management)'.decode('utf-8')
correction['johan van benthem']    = 'johan van benthem (logician)'.decode('utf-8')
correction['springer (company)']   = 'springer publishing'.decode('utf-8')
correction['springer']             = 'springer publishing'.decode('utf-8')
correction['tesla-company']        = 'tesla motors'.decode('utf-8')
correction['tesla, inc.']          = 'tesla motors'.decode('utf-8')
correction['ecole polytechnique']  = u'École polytechnique'
correction['jurgen schmidhuber']   = u'jürgen schmidhuber'
#correction['jürgen schmidhuber']   = 'j\xfcrgen schmidhuber'
correction['agi (disambiguation)'] = 'agi'.decode('utf-8')
correction['graphs']               = 'graph (abstract data type)'.decode('utf-8')
correction['titles in academia']   = 'technical director'.decode('utf-8')
correction['shape analysis']       =      'shape analysis (digital geometry)'.decode('utf-8')
correction['idsia']                =   'dalle molle institute for artificial intelligence research'.decode('utf-8')
correction['zfc set theory']       = 'zermelo–fraenkel set theory'.decode('utf-8')
correction['tokenization']         = 'tokenization (lexical analysis)'.decode('utf-8')
correction['nltk']                 = 'natural language toolkit'.decode('utf-8')
correction['icml']                 = 'international conference on machine learning'.decode('utf-8')
correction['differential equations'] = 'differential equation'.decode('utf-8')
correction['data clustering']      = 'cluster analysis'.decode('utf-8')
correction['prime numbers']        = 'prime number'.decode('utf-8')
correction['ubiquitous robotics']  = 'ubiquitous robot'.decode('utf-8')
correction['the artificial intelligence'] = 'artificial intelligence'.decode('utf-8')
correction['swarm techniques']     = 'swarm intelligence'.decode('utf-8')
correction['deep neural network']  = 'deep learning'.decode('utf-8')
correction['sharing knowledge']    = 'knowledge sharing'.decode('utf-8')
correction['baidu.com inc']                     = 'baidu'.decode('utf-8')
correction['feature (computer vision)']       = 'feature detection (computer vision)'.decode('utf-8')
correction['x'] = 'macos'.decode('utf-8')
correction['the smashing pumpkins 1991–1998'] = 'the smashing pumpkins'.decode('utf-8')
correction['hatsukoi cider / deep mind']      = 'deepmind'.decode('utf-8')
###################################################################################################################
def getDict():
    title2ind, ind2title = {}, {}
    if not os.path.exists('title_map.pkl'):
        lines = codecs.open('title.txt', 'r', encoding = 'utf-8').readlines()
        lines = [x.strip().lower() for x in lines]
        ind2title = dict([(i, x) for i, x in enumerate(lines)])
        title2ind = dict([(line, i) for i, line in enumerate(lines)])
        with open('title_map.pkl', 'w') as fp :
            pickle.dump(title2ind, fp)
            pickle.dump(ind2title,  fp)
    else:
        with open('title_map.pkl', 'r') as fp :
            title2ind = pickle.load(fp)
            ind2title = pickle.load(fp)
    return title2ind, ind2title

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

def cal_tag_tag_sim(tag_pkls, quantile = 95):
    '''
    :param tag_pkls:
    :return:
    #@TODO should also generate the
    '''
    Tag_tag_topk_sims = []
    from collections import defaultdict
    concept2aug = defaultdict(list)

    for tag_pkl in tag_pkls:
        concepts, sims = get_query_result('query_pkl/' + tag_pkl)
        sims = np.array(sims)
        sorted_sim_indices = np.argsort(sims)

        threhold = np.percentile(sims, quantile)

        topk_sim = np.sort(sims[sims > threhold])[::-1]

        topk_quantile_indices = np.nonzero(np.in1d(sims, topk_sim))[0]

        topk_sim_indices = np.sort(topk_quantile_indices)[:-1]

        topk_concepts = np.array(concepts)[topk_sim_indices]

        Tag_tag_topk_sims.append([topk_sim_indices, topk_concepts, topk_sim])

        concept2aug.update({ tag_pkl.split('.')[0] : topk_concepts.tolist() })

    pickle.dump(Tag_tag_topk_sims, open('Tag_tag_sim_topk_{}quantile'.format(str(quantile)) + '.pkl', 'wb'))
    pickle.dump(concept2aug,       open('concept2aug_{}quantile.pkl'.format(str(quantile)), 'wb'))
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
