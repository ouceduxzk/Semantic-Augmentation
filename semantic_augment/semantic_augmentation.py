#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os, sys , pickle
import numpy as np
from util import *
import wikipedia
import matplotlib.pyplot as plt
import glob
from heatmap_plot import Heatmap

from sklearn.metrics.pairwise import cosine_similarity

'''
    Some difficulty of dealing with tags are :
        1. inconsistency mapping from collected data to wiki url
        2. inconsisitency mapping from wiki url concept to the local title.txt (2015 dump) concept
    So whenever there is some errros occuring, need to be corrected manually for the second step.

    Important Notes :

    1. standarize concepts from the tag files
    2. remember the trade-off of cosine-similarity
    3. generate the distribution of frequency of user_concepts and obj_concepts
    4. find one example in the augmented u-u diff plot and look at what are the augmented examples
    5. play with the quantile, which controls how much augmentation we are doing and thus influence the
        self-similarity plot .、

    6 I can put the obj and user in the same matrix and visualize only part of it. (done)

    7.  find the indices for the min and max of this difference matrix and
        for each of such (i, j), find out what is the features used before
        and after the augmentation for the user_or_obj i and j.

    8  print out the similairties for 7, also print out the u/o for each case

    9. (i, j) and (j, i) pairs, calcaute only once

    10 #@TODO for each tag i with user j, save the corresponding augmented tag for building a graph , with the weights
    being the similarity

    11. #TODO find out which tag associate with which augmented concept and then get the value of the corresponding dict
      with the key being each indice in the col of obj_tag matrix

    12  output the edges that have the intersection .

'''
title2ind, ind2title = getDict()
concepts = correction.keys()
for key in concepts:
    title2ind[key] = title2ind[correction[key]]

class SemanticAug(object):
    def get_doc_similarity(self, Tag_tag_matrix, ind):
        tag_ind_vec = Tag_tag_matrix[ind,][0]
        #print(tag_ind_vec)
        indices = np.argsort(tag_ind_vec)[::-1]
        #print(indices )
        values = tag_ind_vec[indices]
        return indices, values
    '''
        semantic augmentation using all tags
    '''
    def semantic_aug_all(self, Tag_tag_matrix, Obj_tag_matrix):
        Aug_new = np.zeros(Obj_tag_matrix.shape)
        for col in range(Obj_tag_matrix.shape[1]):
            print(Tag_tag_matrix.shape, Obj_tag_matrix.shape)
            Aug_new[:, col] = np.sum(np.multiply(Tag_tag_matrix , Obj_tag_matrix[:,col].transpose()), 1)
        print(Aug_new)
        return Aug_new
    '''
        semantic augmentaiton using topk data, k is pre-calculated for each concept
    '''

    def semantic_aug_top_K(self, Tag_tag_sim, Obj_tag_matrix, Obj_tag_indexing, prefix = 'ou_'):
        '''
        :param Tag_tag_sim: a list of [topk_sim_indices, topk_concepts, topk_sim] , each of the entry in the list
            is a list with size k, and k is not constant for different concept
            lenght is #of samples
        :param Tag_obj_matrix: dim is n x #of samples
        :param Obj_tag_indexing : a list of  global index for tags occured in Obj_tag_matrix (0-4m)
        :return: Obj_tag_augment : dim is n x p, where p > 500
        '''
        # step 1 : calcualate the # of columns in Obj_tag_agument matrix
        # col_indices = []
        # for entry in Tag_tag_sim :
        #     topk_sim_indices = entry[0]
        #     col_indices.extend(topk_sim_indices)
        # unique_indices =  list(set(col_indices))
        # col = len(unique_indices) + len(Obj_tag_indexing)

        print('the length of tag_tag_sim is {}'.format(len(Tag_tag_sim) ))
        print('step 1 : calcualate the # of columns in Obj_tag_agument matrix')
        col_concepts = []
        for entry in Tag_tag_sim :
            topk_sim_concepts = entry[1]
            col_concepts.extend(topk_sim_concepts)
        unique_concetps = list(set(col_concepts))

        ################## print out all the concepts in the unaugmented matrix #####################
        # for i,  ind in  enumerate(Obj_tag_indexing):
        #     print(i, ind2title[ind])

        global_indices_topk_sim = [title2ind[x] for x in unique_concetps]
        ###########################################################################################
        #@TODO union does not preserve the sequence of indices when merge with each other
        #global_total_indices = list(set(global_indices_topk_sim).union(set(Obj_tag_indexing)))
        augment_non_exist_indices = set(global_indices_topk_sim).difference(set(Obj_tag_indexing))
        global_total_indices =  Obj_tag_indexing + list(augment_non_exist_indices)
        ###########################################################################################
        col = len(global_total_indices)

        print(len(Obj_tag_indexing))
        print(len(global_total_indices))

        #print([ ind2title[x] for x in set(global_indices_topk_sim).intersection(set(Obj_tag_indexing))])
        print('step 2 : hash each index in the unique_indices')
        global_local_map = dict([(ind, i) for i, ind in enumerate(global_total_indices)])
        local_global_map = dict(enumerate(global_total_indices))

        with open('save/' + prefix + 'local_global_indexing.pkl', 'wb') as fp :
            pickle.dump(global_local_map, fp)
            pickle.dump(local_global_map, fp)

        tmp = []
        for i, x in enumerate(global_total_indices):
            tmp.append((i, ind2title[x]))
            #print(x, global_local_map[x], ind2title[x])

        aug_concept_dict = dict(tmp)
        import operator
        # for key, value in sorted(aug_concept_dict.items(), key=operator.itemgetter(0)):
        #     print(key, value)
        #
        # # the key of ang_concept_dict is local indices 0-n
        # the value of aug_concept_dict is the ttile in string

        pickle.dump(aug_concept_dict, open('save/' + prefix + 'aug_concept_index.pkl', 'wb'))
        print(Obj_tag_matrix.shape)
        print(len(Tag_tag_sim))

        Obj_tag_augment = np.zeros((Obj_tag_matrix.shape[0], col))

        print('step 3 : put the similarity entry for each Tag into the corresponding position of Obj_tag_agument')
        for row in range(Obj_tag_matrix.shape[0]):
            for c in range(Obj_tag_matrix.shape[1]):
                # the cth entry in the Obj_tag_matrix is the one we need
                Obj_tag_augment[row,c]  = Obj_tag_matrix[row, c]
                if Obj_tag_matrix[row, c] == 1:
                    aug = Tag_tag_sim[c]
                    #aug_sims = [x[1] for x in aug]
                    aug_indices, aug_concepts, aug_sims  = aug
                    for sim, concept in zip(aug_sims, aug_concepts):    # [1:]
                        try :
                            global_index = title2ind[concept]
                        except :
                            redirected = wikipedia.page(concept)
                            concept = redirected.title.lower()
                            global_index = title2ind[concept]

                        local_index = global_local_map[global_index]
                        Obj_tag_augment[row, local_index] += sim

        print('the sparsity of un augmented matrix  is {}'.format((Obj_tag_matrix > 0).sum()/1.0/np.prod(Obj_tag_matrix.shape)))
        return Obj_tag_augment

    def get_obj_tag_matrix(self):
        fp = open('save/entity_tag_matrix.pkl', 'rb')
        ou_concepts = pickle.load(fp)
        ou_tag_matrix = pickle.load(fp)
        users_concept = pickle.load(fp)
        user_tag_matrix = pickle.load(fp)
        return users_concept, user_tag_matrix, ou_concepts, ou_tag_matrix

    def test_semantic_aug_topk(self, tag_matrix_pkl, quantile = '_90'):
        Tag_tag_sim = pickle.load(open(tag_matrix_pkl, 'rb'))

        if not os.path.exists('save/oandu_tag_matrix.pkl'):
            user_concepts,  user_tag_matrix, ou_concepts, ou_tag_matrix = self.get_obj_tag_matrix()
            user_tag_indices = [ title2ind[x.encode('utf-8')] for x in user_concepts]
            ou_tag_indices = [ title2ind[x.encode('utf-8')] for x in ou_concepts]

            fp = open('save/oandu_tag_matrix.pkl', 'wb')
            pickle.dump(user_tag_matrix,fp)
            pickle.dump(user_tag_indices, fp)
            pickle.dump(ou_tag_matrix, fp)
            pickle.dump(ou_tag_indices, fp)
        else:
            fp = open('save/oandu_tag_matrix.pkl', 'rb')
            user_tag_matrix  = pickle.load(fp)
            user_tag_indices  = pickle.load(fp)
            ou_tag_matrix  = pickle.load(fp)
            ou_tag_indices = pickle.load(fp)

        ou_tag_aumgent = self.semantic_aug_top_K(Tag_tag_sim, ou_tag_matrix, ou_tag_indices, 'ou_')
        user_tag_augment = self.semantic_aug_top_K(Tag_tag_sim, user_tag_matrix, user_tag_indices, 'usr_')

        fpobj = open('save/augment_all' + quantile + '.pkl', 'wb')
        pickle.dump(ou_tag_matrix, fpobj)
        pickle.dump(ou_tag_aumgent, fpobj)
        pickle.dump(user_tag_matrix, fpobj)
        pickle.dump(user_tag_augment, fpobj)

    def plot_augment(self, quantile = '_90'):
        fpobj = open('save/augment_all' + quantile + '.pkl', 'rb')
        ou_tag_matrix = pickle.load(fpobj)
        ou_tag_aumgent = pickle.load(fpobj)
        User_tag_matrix = pickle.load(fpobj)
        User_tag_augment = pickle.load(fpobj)
        ################## user and obj sim #######################################################
        ou_sim = cosine_similarity(ou_tag_matrix, ou_tag_matrix)
        oaug_oaug_sim = cosine_similarity(ou_tag_aumgent, ou_tag_aumgent)
        oo_diff = oaug_oaug_sim - ou_sim
        oodiff = Heatmap(range(oo_diff.shape[0]), range(oo_diff.shape[1]), oo_diff.tolist())
        oodiff.plot('ou diff', 'oudiff' + quantile + '.png')
        ooplot = Heatmap(range(ou_sim.shape[0]), range(ou_sim.shape[1]), ou_sim.tolist())
        ooplot.plot('ou similarity', 'ousim' + quantile + '.png')
        ooaugplot = Heatmap(range(oaug_oaug_sim.shape[0]), range(oaug_oaug_sim.shape[1]), oaug_oaug_sim.tolist())
        ooaugplot.plot('ou similarity with semantic augmentation', 'ouaug' + quantile + '.png')

        ################# user obj sim ########################################################
        u_u_sim = cosine_similarity(User_tag_matrix, User_tag_matrix)
        u_u_aug_sim = cosine_similarity(User_tag_augment, User_tag_augment)
        u_u_diff =  u_u_aug_sim - u_u_sim
        uudiff = Heatmap(range(u_u_diff.shape[0]), range(u_u_diff.shape[1]), u_u_diff.tolist())
        uudiff.plot('uu diff', 'uudiff' + quantile + '.png')
        uuplot = Heatmap(range(u_u_sim.shape[0]), range(u_u_sim.shape[1]), u_u_sim.tolist())
        uu_augplot = Heatmap(range(u_u_aug_sim.shape[0]), range(u_u_aug_sim.shape[1]), u_u_aug_sim.tolist())
        uuplot.plot('uu similarity', 'uusim'  + quantile + '.png')
        uu_augplot.plot('uu similarity with semantic augmentation', 'uuaug' + quantile + '.png')
        #######################################################################################

        fpsave = open('save/diff' + quantile + '.pkl', 'wb')
        pickle.dump(oo_diff, fpsave)
        pickle.dump(u_u_diff, fpsave)

    def diff_diff(self, quantile1, quantile2):
        fpobj1 = open('save/diff' + quantile1 + '.pkl', 'rb')
        ou_diff1 = pickle.load(fpobj1)
        uu_diff1 = pickle.load(fpobj1)

        fpobj2 = open('save/diff' + quantile2 + '.pkl', 'rb')
        ou_diff2 = pickle.load(fpobj2)
        uu_diff2 = pickle.load(fpobj2)

        ou_ddiff = ou_diff1 - ou_diff2
        ouddiff = Heatmap(range(ou_ddiff.shape[0]), range(ou_ddiff.shape[1]), ou_ddiff.tolist())
        ouddiff.plot('ou ddiff', 'ouddiff' + quantile1 + quantile2 + '.png')


        uu_ddiff = uu_diff1 - uu_diff2
        uuddiff  = Heatmap(range(uu_ddiff.shape[0]), range(uu_ddiff.shape[1]), uu_ddiff.tolist())
        uuddiff.plot('uu ddiff', 'uuddiff' + quantile1 + quantile2 + '.png')


    def largest_indices(self, ary, n):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, -n)[-n:]
        #indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)

    def smallest_indices(self, ary, n):
        """Returns the n largest indices from a numpy array."""
        flat = ary.flatten()
        indices = np.argpartition(flat, n)[:n]
        #indices = indices[np.argsort(-flat[indices])]
        return np.unravel_index(indices, ary.shape)

    def get_tag(self, tag_map, user_obj_map, ind):
        '''
            user_obj_map is
                { ind :  [tag_idx1, tag_idx2]}
            tag_map is:
                { tag_idx :  tag_name}
        '''
        user_or_obj = user_obj_map[ind]
        return [tag_map[x] for x in user_or_obj]

    def entity_similairity_change_after_augm(self, n, prefix = 'ou_', quantile_pkl = 'concept2aug_95quantile.pkl', quantile = 95):
        '''
        %:param mat: the o-u difference matrix
        :param n:   top k smallest and largest elements
        :return: ...
        '''
        fpobj = open('save/augment_all_{}.pkl'.format(str(quantile)), 'rb')
        ou_tag_matrix = pickle.load(fpobj)
        ou_tag_aumgent = pickle.load(fpobj)
        User_tag_matrix = pickle.load(fpobj)
        User_tag_augment = pickle.load(fpobj)

        u_u_sim = cosine_similarity(ou_tag_matrix, ou_tag_matrix)
        u_u_aug_sim = cosine_similarity(ou_tag_aumgent, ou_tag_aumgent)
        mat = u_u_aug_sim - u_u_sim

        larg_indices = self.largest_indices(mat, n)
        sma_indices = self.smallest_indices(mat, n)

        n_large_row = len(larg_indices[0])
        n_large_col = len(larg_indices[1])
        n_sma_row = len(sma_indices[0])
        n_sma_col = len(sma_indices[1])

        rows = larg_indices[0].tolist() + sma_indices[0].tolist()
        cols = larg_indices[1].tolist()+ sma_indices[1].tolist()

        augtag2ind  = pickle.load(open('save/' + prefix + 'aug_concept_index.pkl', 'rb'))
        fp = open(quantile_pkl, 'rb')
        concept2aug = pickle.load(fp)
        concept2aug_sim = pickle.load(fp)

        for row, col in zip(rows, cols):
            #row_name = user_obj_names[row]
            #col_name = user_obj_names[col]
            tags_for_one_row = ou_tag_matrix[row,:].toarray()
            tags_row_aug     = ou_tag_aumgent[row,:]
            indices_row = np.where(tags_for_one_row > 0)[1].tolist()

            indices_row_aug = np.where(tags_row_aug > 0 )[0].tolist()

            tags_for_one_col = ou_tag_matrix[col,:].toarray()
            tags_col_aug     = ou_tag_aumgent[col,:]

            indices_col = np.where(tags_for_one_col > 0)[1].tolist()
            indices_col_aug = np.where(tags_col_aug > 0)[0].tolist()

            tags_for_row = [ augtag2ind[x] for x in indices_row]
            tags_for_col = [ augtag2ind[x] for x in indices_col]

            tags_for_row_aug = [augtag2ind[x] for x in indices_row_aug]
            tags_for_col_aug = [augtag2ind[x] for x in indices_col_aug]

            intersected_words = self.get_intersection_words(concept2aug, tags_for_row_aug, tags_for_col_aug)
            print(np.where(tags_for_one_row > 0))
            print(np.where(tags_for_one_col > 0))

            print intersected_words
            print(len(intersected_words))
            fp = open('gephi_{}_{}_{}_{}.csv'.format(row, col, u_u_sim[row,col], u_u_aug_sim[row,col]), 'wb')
            fp.write('source,target,similarity\n')
            for key in tags_for_row:
                fp.write('researcher1' + ',' + key + ',1\n')
                for j, (value, sim) in enumerate(zip(concept2aug[key], concept2aug_sim[key])):
                    #print(key, value, sim)
                    if value in intersected_words:
                        fp.write(key + ','  + value + ',' + str(sim) + '\n')
                    # if j > 5 :
                    #     break

            for i, key in enumerate(tags_for_col):
                fp.write('researcher2'  + ',' + key + ',1\n')
                for j, (value, sim) in enumerate(zip(concept2aug[key], concept2aug_sim[key])):
                    if value in intersected_words:
                        fp.write(key + ',' + value + ',' + str(sim) + '\n')
                    # if j > 5 :
                    #     break
            fp.close()
            print('-----------------------------------------')
            print('the indices of two entities out of 73 is {} and {}'.format(row, col))
            print('the similarity of two entities before and after is {} and {}'.format(u_u_sim[row,col], u_u_aug_sim[row,col] ))
            print(tags_for_row, tags_for_col)
            print(tags_for_row_aug, tags_for_col_aug)
            print('-----------------------------------------')

    def get_intersection_words(self, concept2aug, c1, c2):
        l1 = []
        l2 = []
        for key in c1 :
            l1.extend(concept2aug[key])
        for key in c2 :
            l2.extend(concept2aug[key])

        intersect = set(l1).intersection(set(l2))
        return intersect

    def test_concept_aug(self, pkl):
        # the first element of the augmented tag is the same, should be eliminated.
        concept2aug = pickle.load(open(pkl , 'rb'))
        print(concept2aug)


if __name__ == '__main__':
    obj = SemanticAug()
    #obj.test_concept_aug('concept2aug_95quantile.pkl')

    #obj.test_semantic_aug_topk('Tag_tag_sim_topk_80quantile.pkl', '_80')
    #obj.test_semantic_aug_topk('Tag_tag_sim_topk_90quantile.pkl', '_90')
    obj.test_semantic_aug_topk('Tag_tag_sim_topk.pkl', '_95')
    #Obj_tag_indices, User_tag_indices = obj.get_concept_indices()
    #obj.plot_augment('_95')
    #obj.plot_augment('_80')
    #obj.diff_diff('_80', '_95')
    obj.entity_similairity_change_after_augm(10)
    # data = pickle.load(open('obj_tag.pkl', 'rb'))
    # #plt.imshow(data, cmap='hot', interpolation='nearest')
    # print(data[0,:])
    # aug_data = pickle.load(open('augment_obj_tag.pkl', 'rb'))
    # # plt.imshow(aug_data, cmap = 'hot', interpolation='nearest')
    # # plt.show()
    # print(aug_data[0,:])
    # indices = np.where(aug_data[0,:] > 0)[0].tolist()
    # #index_concept = pickle.load(open('aug_concept_index.pkl', 'rb'))
    # fp = open('save/local_global_indexing.pkl', 'rb')
    # global_local_indexing = pickle.load(fp)
    # local_global_indexing = pickle.load(fp)
    # print([ ind2title[local_global_indexing[x]] for x in indices])
    # # print(index_concept)
    # # for ind in indices :
    # #     print(index_concept[ind])

    #340
    #9091