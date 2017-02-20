import numpy as np
import os, pickle
import glob
from sklearn.metrics.pairwise import cosine_similarity

pkls = glob.glob('sp_tfidf/*pkl')
pkls = sorted(pkls, key = lambda v: int(v.split('/')[-1].split('.')[0].split('sp_')[1]))
doc_tfidfs = [pickle.load(open(pkl, 'r')) for pkl in pkls]

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

def getIndexOfPklFile(pkls, ind):
    doc_numbers = [ int(x.split('/')[-1][3:].split('.')[0]) for x in pkls ]
    for i, doc_n in enumerate(doc_numbers):
        if doc_n > ind :
            return i
    return 0

def cal_sim_topn(pkls, i, ind, ind2title, topn = 1000):
    concept = ind2title[ind]
    doc_ids = [ int(x.split('/')[-1][3:].split('.')[0]) for x in pkls ]
    #print(doc_ids)
    src_tfidf = doc_tfidfs[i][ind,:]
    for i, (doc_tfidf, doc_id) in enumerate(zip(doc_tfidfs, doc_ids)):
        if i == 0 :
            total_sim = cosine_similarity(src_tfidf, doc_tfidf)
        else:
            part_sim = cosine_similarity(src_tfidf , doc_tfidf[doc_ids[i-1]:,:])
            #print(total_sim.shape, part_sim.shape)
            total_sim = np.hstack([total_sim, part_sim])
            print(part_sim.shape)

    sim_dec_order = np.argsort(total_sim[0, :])[::-1]
    top_n_indices = sim_dec_order[:topn].tolist()
    top_n_sims = [ total_sim[0][x] for x in top_n_indices]
    #print(top_n_indices)
    top_n_concepts = [ ind2title[x] for x in top_n_indices]

    #total_concepts = [ ind2title[x] for x in sim_dec_order]
    pickle.dump(zip(top_n_concepts, top_n_sims), open(concept + '.pkl', 'rb'))
    print('the top {} concepts similiar to {} is'.format(str(topn),  concept))
    print(zip(top_n_concepts, top_n_sims))

def doc_sim_from_index(ind, top_n):
    title2ind, ind2title = getDict()
    #pkls = glob.glob('sp_tfidf/*pkl')
    #pkls = sorted(pkls, key=lambda v: int(v.split('/')[-1].split('.')[0].split('sp_')[1]))
    ind_pkl = getIndexOfPklFile(pkls, ind)
    return cal_sim_topn(pkls, ind_pkl, ind, ind2title, 1+ top_n)[1][1]

def doc_sim_by_cosine(concept):
    most_sim_concepts = []
    title2ind, ind2title = getDict()
    ind = title2ind[concept]
    ind_pkl = getIndexOfPklFile(pkls, ind)
    cal_sim_topn(pkls, ind_pkl, ind, ind2title)
    return most_sim_concepts

if __name__ == '__main__':
    title2ind, ind2title = getDict()
    random_list_concepts = np.random.choice(title2ind.keys(), 1000)

    for query in random_list_concepts:
        doc_sim_by_cosine(query)