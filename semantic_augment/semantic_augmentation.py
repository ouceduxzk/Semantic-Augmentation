import os, sys , pickle
import numpy as np
from util import *
import matplotlib.pyplot as plt

title2ind, ind2title = getDict()

def get_doc_similarity(Tag_tag_matrix, ind):
    tag_ind_vec = Tag_tag_matrix[ind,][0]
    #print(tag_ind_vec)
    indices = np.argsort(tag_ind_vec)[::-1]
    #print(indices )
    values = tag_ind_vec[indices]
    return indices, values
'''
    semantic augmentation using all tags
'''
def semantic_aug_all(Tag_tag_matrix, Obj_tag_matrix):
    Aug_new = np.zeros(Obj_tag_matrix.shape)
    for col in range(Obj_tag_matrix.shape[1]):
        print(Tag_tag_matrix.shape, Obj_tag_matrix.shape)
        Aug_new[:, col] = np.sum(np.multiply(Tag_tag_matrix , Obj_tag_matrix[:,col].transpose()), 1)
    print(Aug_new)
    return Aug_new

'''
    semantic augmentaiton using topk data, k is pre-calculated for each concept
'''

def semantic_aug_top_K(Tag_tag_sim, Obj_tag_matrix, Obj_tag_indexing):
    '''
    :param Tag_tag_sim: a list of [topk_sim_indices, topk_concepts, topk_sim] , each of the entry in the list
            is a list with size k, and k is not constant for different concept
            lenght is #of samples

    :param Tag_obj_matrix: dim is n x #of samples

    :param Obj_tag_indexing : a list of indices of tags in the Obj_tag_matrix (0-4m)

    :return: Obj_tag_augment : dim is n x p, where p > 500
    '''
    # step 1 : calcualate the # of columns in Obj_tag_agument matrix
    col_concepts = []

    for entry in Tag_tag_sim :
        #topk_sim_indices = entry[0]
        topk_sim_concepts = entry[1]
        col_concepts.extend(topk_sim_concepts)

    unique_concetps =  list(set(col_concepts))
    global_indices_topk_sim = [ title2ind[x] for x in unique_concetps]
    total_indices = list(set(global_indices_topk_sim).union(set(Obj_tag_indexing)))

    col = len(total_indices)

    #print(set(global_indices_topk_sim).intersection(set(Obj_tag_indexing)))
    #print([ ind2title[x] for x in set(global_indices_topk_sim).intersection(set(Obj_tag_indexing))])

    # step 2 : hash each index in the unique_indices

    global_local_map = dict([(ind, i) for i, ind in enumerate(total_indices)])
    local_global_map = dict(enumerate(total_indices))

    with open('local_global_indexing.pkl', 'wb') as fp :
        pickle.dump(global_local_map, fp)
        pickle.dump(local_global_map, fp)

    # indexing_idx = {}
    # for i , e in enumerate(Obj_tag_indexing):
    #     indexing_idx.update({gloabal_local_map[e]: i})
    #
    # for i,  e in enumerate(global_indices_topk_sim):
    #     indexing_idx.update({e : i + len(Obj_tag_indexing)})

    # save the dictionary of  index :  concept in the augmented obj-tag matrix

    # aug_concept_dict = {}
    # for i in range(len(Obj_tag_indexing)):
    #     concept = ind2title[Obj_tag_indexing[i]]
    #     aug_concept_dict.update({ i : concept})
    #
    # for i in range(len(global_indices_topk_sim)):
    #     concept = ind2title[ global_indices_topk_sim[i] ]
    #     aug_concept_dict.update({ i + len(Obj_tag_indexing) : concept})

    aug_concept_dict = dict([(global_local_map[x] ,ind2title[x]) for x in total_indices])
    pickle.dump(aug_concept_dict, open('aug_concept_index.pkl', 'wb'))
    #print(aug_concept_dict)
    Obj_tag_augment = np.zeros((len(Tag_tag_sim), col))


    # step 3 : put the similarity entry for each Tag into the corresponding position of Obj_tag_agument

    for row in range(Obj_tag_matrix.shape[0]):
        for c in range(Obj_tag_matrix.shape[1]):
            # the cth entry in the Obj_tag_matrix is the one we need
            Obj_tag_augment[row,c]  = Obj_tag_matrix[row, c]

            if Obj_tag_matrix[row, c] == 1:
                aug = Tag_tag_sim[c]
                aug_indices, aug_concepts, aug_sims  = aug
                for sim, concept in zip(aug_sims[1:], aug_concepts[1:]):
                    global_index = title2ind[concept]
                    local_index = global_local_map[global_index]
                    Obj_tag_augment[row, local_index] += sim

    return Obj_tag_augment

def get_concept_indices():
    #title2ind, ind2title = getDict()
    concepts = get_concept_samples()
    return [ title2ind[concept] for concept in concepts ]

def test_3x3():
    #T_tag_matrix = np.array([[1, 0.2, 0.3, 0.5],[0.4, 1.0, 0.7, 0.3], [0.2, 0.3, 1.0, 0.5]])
    T_tag_matrix = np.array([[1, 0.2, 0.3], [ 0.4, 1.0, 0.7]])
    T_obj_matrix = np.array([[1, 0, 0], [0,0, 1], [1, 1, 0]]).astype(np.float)
    print('the TT matrix is: ')
    print(T_tag_matrix)
    print(T_obj_matrix)
    semantic_aug_all(T_tag_matrix, T_obj_matrix)

def test_semantic_aug_topk():
    Tag_tag_sim = pickle.load(open('Tag_tag_sim_topk.pkl', 'rb'))
    np.random.random(520)
    Obj_tag_matrix = np.zeros((2, len(Tag_tag_sim)))
    Obj_tag_matrix[0,1] = 1

    #(np.random.random((5, len(Tag_tag_sim))) > 0.6).astype(np.float)
    Obj_tag_indices = get_concept_indices()
    Obj_tag_aumgent = semantic_aug_top_K(Tag_tag_sim, Obj_tag_matrix, Obj_tag_indices)
    pickle.dump(Obj_tag_aumgent, open('augment_obj_tag.pkl', 'wb'))
    pickle.dump(Obj_tag_matrix, open('obj_tag.pkl', 'wb'))

def testa():
    data = pickle.load(open('Tag_tag_sim_topk.pkl', 'rb'))
    for item in data :
        print(item[1])

def check_ml():
    #data = pickle.load(open('query_pkl/Machine learning.pkl', 'rb'))
    concepts, sims = get_query_result('query_pkl/A.I. Artificial Intelligence.pkl')
    print(zip(concepts, sims))
    # sims = np.array(sims)
    # sorted_sim_indices = np.argsort(sims)
    # #print(concepts, sims)
    # threhold = np.percentile(sims, 95)
    #
    # print(threhold)
    #
    # topk_sim = np.sort(sims[sims > threhold])[::-1]
    #
    # topk_quantile_indices = np.nonzero(np.in1d(sims, topk_sim))[0]
    #
    # topk_sim_indices = np.sort(topk_quantile_indices)[:-1]
    #
    # topk_concepts = np.array(concepts)[topk_sim_indices]
    # print(topk_sim_indices, topk_concepts, topk_sim)

if __name__ == '__main__':
    check_ml()

    # test_semantic_aug_topk()
    # data = pickle.load(open('obj_tag.pkl', 'rb'))
    #
    # #plt.imshow(data, cmap='hot', interpolation='nearest')
    # #plt.show()
    #
    # print(data[0,:])
    #
    # aug_data = pickle.load(open('augment_obj_tag.pkl', 'rb'))
    # # plt.imshow(aug_data, cmap = 'hot', interpolation='nearest')
    # # plt.show()
    #
    # print(aug_data[0,:])
    #
    # indices = np.where(aug_data[0,:] > 0)[0].tolist()
    # #index_concept = pickle.load(open('aug_concept_index.pkl', 'rb'))
    #
    # fp = open('local_global_indexing.pkl', 'rb')
    # global_local_indexing = pickle.load(fp)
    # local_global_indexing = pickle.load(fp)
    #
    # print([ ind2title[local_global_indexing[x]] for x in indices])
    #
    # # print(index_concept)
    # # for ind in indices :
    # #     print(index_concept[ind])
    #
    # # data = pickle.load(open('concept_AI.pkl', 'rb'))
    # # for item in data :
    # #     ind, concepts, sims = item
    # #     print(concepts[:10])
