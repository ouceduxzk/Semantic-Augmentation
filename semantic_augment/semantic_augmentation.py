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
    user are 25 ...
    result is user-user, user-object matrix
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

    def semantic_aug_top_K(self, Tag_tag_sim, Obj_tag_matrix, Obj_tag_indexing):
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

        print('step 1 : calcualate the # of columns in Obj_tag_agument matrix')
        col_concepts = []
        for entry in Tag_tag_sim :
            topk_sim_concepts = entry[1]
            col_concepts.extend(topk_sim_concepts)
        unique_concetps = list(set(col_concepts))

        global_indices_topk_sim = [title2ind[x] for x in unique_concetps]
        global_total_indices = list(set(global_indices_topk_sim).union(set(Obj_tag_indexing)))
        col = len(global_total_indices)

        #print(set(global_indices_topk_sim).intersection(set(Obj_tag_indexing)))
        #print([ ind2title[x] for x in set(global_indices_topk_sim).intersection(set(Obj_tag_indexing))])
        print('step 2 : hash each index in the unique_indices')
        global_local_map = dict([(ind, i) for i, ind in enumerate(global_total_indices)])
        local_global_map = dict(enumerate(global_total_indices))

        with open('save/local_global_indexing.pkl', 'wb') as fp :
            pickle.dump(global_local_map, fp)
            pickle.dump(local_global_map, fp)

        # aug_concept_dict = {}
        # for i in range(len(Obj_tag_indexing)):
        #     concept = ind2title[Obj_tag_indexing[i]]
        #     aug_concept_dict.update({ i : concept})
        # for i in range(len(global_indices_topk_sim)):
        #     concept = ind2title[ global_indices_topk_sim[i] ]
        #     aug_concept_dict.update({ i + len(Obj_tag_indexing) : concept})
        tmp = []
        for x in global_total_indices:
            tmp.append((global_local_map[x], ind2title[x]))
        aug_concept_dict = dict(tmp)
        pickle.dump(aug_concept_dict, open('save/aug_concept_index.pkl', 'wb'))
        #@TODO bug (fixed) : the dimension of Obj_tag_augment should be (# of user/objs, col)
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
                    for sim, concept in zip(aug_sims[1:], aug_concepts[1:]):
                        try :
                            global_index = title2ind[concept]
                        except :
                            redirected = wikipedia.page(concept)
                            concept = redirected.title.lower()
                            global_index = title2ind[concept]

                        local_index = global_local_map[global_index]
                        Obj_tag_augment[row, local_index] += sim

        return Obj_tag_augment

    def get_obj_tag_matrix(self):
        fp = open('save/entity_tag_matrix.pkl', 'rb')
        #obj_tag_matrix = pickle.load(fp)
        users_concept = pickle.load(fp)
        user_tag_matrix = pickle.load(fp)
        return users_concept, user_tag_matrix

    def test_semantic_aug_topk(self):
        Tag_tag_sim = pickle.load(open('Tag_tag_sim_topk.pkl', 'rb'))

        if not os.path.exists('save/oandu_tag_matrix.pkl'):
            user_concepts,  User_tag_matrix = self.get_obj_tag_matrix()
            User_tag_indices = [ title2ind[x.encode('utf-8')] for x in user_concepts]

            fp = open('save/oandu_tag_matrix.pkl', 'wb')
            #pickle.dump(Obj_tag_matrix,fp)
            pickle.dump(User_tag_matrix, fp)
            #pickle.dump(Obj_tag_indices, fp)
            pickle.dump(User_tag_indices, fp)
        else:
            fp = open('save/oandu_tag_matrix.pkl', 'rb')
            #Obj_tag_matrix  = pickle.load(fp)
            User_tag_matrix  = pickle.load(fp)
            #Obj_tag_indices  = pickle.load(fp)
            User_tag_indices =pickle.load(fp)

        #Obj_tag_aumgent = self.semantic_aug_top_K(Tag_tag_sim, Obj_tag_matrix, Obj_tag_indices)
        User_tag_augment = self.semantic_aug_top_K(Tag_tag_sim, User_tag_matrix, User_tag_indices)

        fpobj = open('save/augment_all.pkl', 'wb')
        #pickle.dump(Obj_tag_matrix, fpobj)
        #pickle.dump(Obj_tag_aumgent, fpobj)
        pickle.dump(User_tag_matrix, fpobj)
        pickle.dump(User_tag_augment, fpobj)

    def plot_augment(self):
        fpobj = open('save/augment_all.pkl', 'rb')
        #Obj_tag_matrix = pickle.load(fpobj)
        #Obj_tag_augment = pickle.load(fpobj)
        User_tag_matrix = pickle.load(fpobj)
        User_tag_augment = pickle.load(fpobj)
        ######################################################################################
        # print(Obj_tag_matrix.shape, Obj_tag_augment.shape)
        # print(User_tag_matrix.shape, User_tag_augment.shape)
        # with open('save/local_global_indexing.pkl', 'rb') as fp :
        #     global_local_map = pickle.load(fp)
        #     local_global_map = pickle.load(fp)
        ################## obj obj sim #######################################################
        # o_o_sim = cosine_similarity(Obj_tag_matrix, Obj_tag_matrix)
        # oaug_oaug_sim = cosine_similarity(Obj_tag_augment, Obj_tag_augment)
        # oo_diff = oaug_oaug_sim - o_o_sim
        # oodiff = Heatmap(range(oo_diff.shape[0]), range(oo_diff.shape[1]), oo_diff.tolist())
        # oodiff.plot('oo diff', 'oodiff.png')
        # ooplot = Heatmap(range(o_o_sim.shape[0]), range(o_o_sim.shape[1]), o_o_sim.tolist())
        # ooplot.plot('o_o similarity', 'oosim.png')
        # ooaugplot = Heatmap(range(oaug_oaug_sim.shape[0]), range(oaug_oaug_sim.shape[1]), oaug_oaug_sim.tolist())
        # ooaugplot.plot('o o similarity with semantic augmentation', 'ooaug.png')
        ################# user obj sim ########################################################
        u_u_sim = cosine_similarity(User_tag_matrix, User_tag_matrix)
        u_u_aug_sim = cosine_similarity(User_tag_augment, User_tag_augment)
        u_u_diff =  u_u_aug_sim - u_u_sim
        uudiff = Heatmap(range(u_u_diff.shape[0]), range(u_u_diff.shape[1]), u_u_diff.tolist())
        uudiff.plot('uu diff', 'uudiff.png')
        ouplot = Heatmap(range(u_u_sim.shape[0]), range(u_u_sim.shape[1]), u_u_sim.tolist())
        ou_augplot = Heatmap(range(u_u_aug_sim.shape[0]), range(u_u_aug_sim.shape[1]), u_u_aug_sim.tolist())
        ouplot.plot('u u similarity', 'uusim.png')
        ou_augplot.plot('u u similarity with semantic augmentation', 'uuaug.png')
        #######################################################################################

if __name__ == '__main__':
    obj = SemanticAug()
    #obj.test_semantic_aug_topk()
    #Obj_tag_indices, User_tag_indices = obj.get_concept_indices()
    obj.plot_augment()
    #@TODO stistical, for rorw, small and largest values
    #                 the histgoram of changement of values 
    # case study of those extreme cases and draw the corresponding graph .
    # data = pickle.load(open('obj_tag.pkl', 'rb'))
    #
    # #plt.imshow(data, cmap='hot', interpolation='nearest')
    #
    # print(data[0,:])
    #
    # aug_data = pickle.load(open('augment_obj_tag.pkl', 'rb'))
    # # plt.imshow(aug_data, cmap = 'hot', interpolation='nearest')
    # # plt.show()
    # print(aug_data[0,:])
    #
    # indices = np.where(aug_data[0,:] > 0)[0].tolist()
    # #index_concept = pickle.load(open('aug_concept_index.pkl', 'rb'))
    #
    # fp = open('save/local_global_indexing.pkl', 'rb')
    # global_local_indexing = pickle.load(fp)
    # local_global_indexing = pickle.load(fp)
    #
    # print([ ind2title[local_global_indexing[x]] for x in indices])
    #
    # # print(index_concept)
    # # for ind in indices :
    # #     print(index_concept[ind])