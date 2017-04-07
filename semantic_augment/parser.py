#!/usr/bin/env python
# -*- coding: utf-8 -*-
import json, os, sys
import numpy as np
import pprint
import matplotlib.pyplot as plt
import operator
import scipy.sparse as sps
import pickle
from util import getDict
import wikipedia
import glob
import pprint
from collections import defaultdict
from collections import Counter
from sklearn.metrics.pairwise import cosine_similarity
from util import correction

class ParseTagEntity(object):
    def parse_tag(self):
        lines = open('collected_data/pandodb_testing-tags')
        objs = []
        users = []
        for line in lines :
            data = json.loads(line)
            tag_name = data['name']
            isUser = True
            tag_id = data['_id']['$oid']
            wiki_url = data['concept_url'].split('wiki/')[-1].replace('_', ' ').lower()

            if data['entity_url'].startswith('/public'):
                entity_id = data['entity_url'].split('/')[-1]
                users.append([ entity_id, wiki_url])
            else:
                entity_id = data['entity_url']
                isUser = False
                objs.append([entity_id, wiki_url])

        user_concepts = list(set([x[1] for x in users]))

        # standardize the concept 'scientific computing' to 'computational science'
        o2omap = pickle.load(open('o2omap_words.pkl', 'rb'))

        user_entities = [x[0] for x in users]
        user_entities = [k for k, v in Counter(user_entities).iteritems() if v > 4]
        users = [[a, o2omap[b] ] for a, b in users if a in user_entities]

        # only those are users have the tag_id in the pandodb_testing-tag_votes database.
        return objs, users

    def parse_obj_tag_matrix(self):
        objs, users = self.parse_tag()
        # obj_concepts = list(set([ x[1] for x in objs ]))
        # #obj_concepts = [k for k, v in Counter(obj_concepts).iteritems() if v > 1 ]
        #
        # obj_entities = [ x[0] for x in objs]
        # obj_entities = [k for k, v in Counter(obj_entities).iteritems() if v > 4]

        concepts = list(set([ x[1] for x in users]))
        entities = list(set([ x[0] for x in users]))

        print('# of user concepts {}'.format(len(concepts)))
        print('# of user entities {}'.format(len(entities)))

        #@TODO build the obj/user and Tag matrix
        '''
            The local_tag_indexing is a map of all concepts to the local index
            The local_entity_indexing is a map of all entities to the local index while
            reverse_entity_indexing the the reverse map of local_index to the name of entity.
        '''
        # local_obj_indexing  = dict([ (item,idx) for idx, item in enumerate(set([x[1] for x in objs])) ])
        # local_user_indexing = dict([ (item,idx) for idx, item in enumerate(set([x[1] for x in users])) ])
        local_tag_indexing  = dict([ (item,idx) for idx, item in enumerate(concepts) ])
        local_entity_indexing = dict([ (item,idx) for idx, item in enumerate(entities) ])

        # reverse_obj_indexing = dict([ (idx, item) for idx, item in enumerate(set([x[1] for x in objs])) ])
        # reverse_user_indexing =dict([ (idx, item) for idx, item in enumerate(set([x[1] for x in users]))])
        reverse_tag_indexing = dict([ (idx, item) for idx, item in enumerate(concepts)])
        reverse_entity_indexing = dict([(idx, item) for idx, item in enumerate(entities)])
        #
        # print('# of objs in o_t matrix is {}'.format(len(local_obj_indexing)))
        # print('# of user in u_t matrix is {}'.format(len(local_user_indexing)))
        print('# of tag in  o_t matrix is {}'.format(len(local_tag_indexing)))
        print('the tags  are {}'.format(local_tag_indexing.keys()))

        fp = open('save/local_indexing.pkl', 'wb')
        # pickle.dump(local_obj_indexing, fp)
        # pickle.dump(local_user_indexing, fp)
        pickle.dump(local_tag_indexing, fp)
        pickle.dump(local_entity_indexing, fp)
        # pickle.dump(reverse_obj_indexing, fp)
        # pickle.dump(reverse_user_indexing, fp)
        pickle.dump(reverse_tag_indexing, fp)
        pickle.dump(reverse_entity_indexing,fp)

        obj_row = []
        obj_col = []
        users_row = []
        users_col = []

        total_row = []
        total_col = []
        # for concept, entity in objs :
        #     obj_row.append(local_obj_indexing[entity])
        #     obj_col.append(local_tag_indexing[concept])

        # for concept, entity in users :
        #     users_row.append(local_user_indexing[entity])
        #     users_col.append(local_tag_indexing[concept])

        for key, value in local_tag_indexing.items():
            if key.startswith('scien'):
                print(key, value)

        for entity, concept  in users :
            total_row.append(local_entity_indexing[entity])
            total_col.append(local_tag_indexing[concept])

        # obj_tag_matrix = sps.coo_matrix(([1] * len(objs), (obj_row, obj_col)), shape=(len(local_obj_indexing), len(local_tag_indexing))).tocsr()
        # user_tag_matrix = sps.coo_matrix(([1] * len(users), (users_row, users_col)), shape=(len(local_user_indexing), len(local_tag_indexing))).tocsc()

        print('the shape of entity_tag matrix is row={}, col={}'.format(len(local_entity_indexing), len(local_tag_indexing)))
        print(total_row)
        print(total_col)
        entity_tag_matrix = sps.coo_matrix(([1] * len(users), (total_row, total_col)), shape=(len(local_entity_indexing), len(local_tag_indexing))).tocsr()
        print('shape of o_t matrix is')
        print(entity_tag_matrix.shape)
        fp = open('save/entity_tag_matrix.pkl', 'wb')
        # pickle.dump(obj_tag_matrix, fp)
        # pickle.dump(user_tag_matrix, fp)
        pickle.dump(concepts, fp)
        pickle.dump(entity_tag_matrix, fp)

    def normalizeWikiTitle(self, list_concepts):
        title2ind, ind2title = getDict()
        ################ correction of keys ###############
        concepts = correction.keys()
        for key in concepts:
            print(key)
            title2ind[key] = title2ind[correction[key]]
            ###################################################
        normalized_concepts = {}
        list_concepts = list(set(list_concepts))
        for i, concept in enumerate(list_concepts):
            print('{} out of {} concepts processed'.format(i, len(list_concepts)))
            tmp = concept.split('wiki/')[-1].lower()
            tmp = tmp.replace('_', ' ')
            if tmp not in title2ind.keys():
                redirected = wikipedia.page(tmp)
                online = redirected.title.lower()
            else:
                online = tmp
            normalized_concepts.update({tmp : online })

        pickle.dump(list_concepts, open('list_concepts.pkl', 'wb'))
        print(len(set(normalized_concepts.keys())), len(set(normalized_concepts.values()))) #408, 379
        pickle.dump(normalized_concepts, open('o2omap_words.pkl', 'wb'))

    # def plot_parse_tag(self):
    #     result, user_result = self.parse_tag()
    #     tag_names = [ x[0] for x in result.values() + user_result.values()]
    #
    #     obj_concepts = [ x[2].split('wiki/')[-1].replace('_', ' ').lower() for x in result.values()]
    #     user_concepts =[ x[1] for in user_result]
    #
    #     pickle.dump(list(set(obj_concepts)), open('save/obj_concepts.pkl', 'wb'))
    #     pickle.dump(list(set(user_concepts)), open('save/user_concepts.pkl', 'wb'))
    #
    #     # print(obj_concepts)
    #     # print(user_concepts)
    #
    #     #entity_id = [ x[1] for x in result + user_result]
    #     list_concepts    = [ x[2] for x in result + user_result]
    #
    #     # print('# of tags : {}'.format(len(tag_names)))
    #     # print('# of unique tags : {}'.format(len(set(tag_names))))
    #     print('# of concepts : {}'.format(len(list_concepts)))
    #     print('# of unique concepts : {}'.format(len(set(list_concepts))))
    #     self.normalizeWikiTitle(list_concepts)
    #     return True


    def plot_sparity(self):
        fp = open('save/entity_tag_matrix.pkl', 'rb')
        obj_tag_matrix = pickle.load(fp)
        user_tag_matrix = pickle.load(fp)

        print('# of non-zero entry of obj_tag_matrix is {}'.format(obj_tag_matrix.nnz))
        print('# of total elemetns of obj_tag_matrix is {}'.format(obj_tag_matrix.shape[0] * obj_tag_matrix.shape[1]))

        print('# of non-zero entry of user_tag_matrix is {}'.format(user_tag_matrix.nnz))
        print('# of total elemetns of user_tag_matrix is {}'.format(user_tag_matrix.shape[0] * user_tag_matrix.shape[1]))
        return True

    def show_json_structure(self, fn):
        lines = open(fn, 'rb').readlines()
        data = json.loads(lines[0])
        print json.dumps(data, indent=4, sort_keys=True)

if __name__ == '__main__':
    import parser
    obj = parser.ParseTagEntity()

    #@TODO regenerate the concepts that are added recently
    
    # fns = glob.glob('collected_data/*')
    # for fn in fns :
    #     print fn
    #     obj.show_json_structure(fn)

    #obj.plot_parse_tag()
    obj.parse_obj_tag_matrix()

    # obj.create_similarity_matrix()
    # obj.plot_sparity()
    # title2ind = {}
    # title2ind['data classification'] = title2ind.pop('data classification (data management)')
    # title2ind['johan van benthem'] = title2ind.pop('johan van benthem (logician)')
    # title2ind['springer (company)'] = title2ind.pop('springer')
    # title2ind['tesla-company'] = title2ind.pop('tesla motors')
    # title2ind['ecole polytechnique'] = title2ind.pop('École polytechnique')
    # title2ind['jurgen schmidhuber'] = title2ind.pop('jürgen schmidhuber')
    # title2ind['agi (disambiguation)'] = title2ind.pop('agi')
    # title2ind['graphs'] = title2ind.pop('graph (abstract data type)')
    # title2ind['titles in academia'] = title2ind.pop('technical director')
    # title2ind['shape analysis'] = title2ind.pop('shape analysis (digital geometry)')
    # title2ind['idsia']          = title2ind.pop('dalle molle institute for artificial intelligence research')
    # title2ind['zfc set theory'] = title2ind.pop('zermelo–fraenkel set theory')
    # title2ind['tokenization'] = title2ind['tokenization (lexical analysis)']
    # title2ind['nltk']         = title2ind['natural language toolkit']
    # title2ind['icml']        =  title2ind['international conference on machine learning']
    # correction = {}
    # correction['data classification'] = 'data classification (data management)'
    # correction['johan van benthem']   = 'johan van benthem (logician)'
    # correction['springer (company)']  = 'springer'
    # correction['tesla-company']       = 'tesla motors'
    # correction['ecole polytechnique'] = 'École polytechnique'
    # correction['jurgen schmidhuber']  = 'jürgen schmidhuber'
    # correction['agi (disambiguation)'] = 'agi'
    # correction['graphs']              = 'graph (abstract data type)'
    # correction['titles in academia']  = 'technical director'
    # correction['shape analysis'] =      'shape analysis (digital geometry)'
    # correction['idsia']             =   'dalle molle institute for artificial intelligence research'
    # correction['zfc set theory'] = 'zermelo–fraenkel set theory'
    # correction['tokenization']   = 'tokenization (lexical analysis)'
    # correction['nltk'] = 'natural language toolkit'
    # correction['icml'] = 'international conference on machine learning'
    # concepts = correction.keys()
    #
    # my_dict = dict((y,x) for x,y in correction.iteritems())
    #
    # ind = 0
    #
    # if ind2title[ind] in  my_dict.keys():
    #     concept = my_dict[ind2title[ind]]
    # from collections import Counter
    # collected_concepts = pickle.load(open('collected_concepts.pkl', 'rb'))
    # list_concepts =  pickle.load(open('list_concepts.pkl', 'rb'))
    # list_concepts = [ x.split('/')[-1].replace('_', ' ') for x in list_concepts]
    # #collected_concepts = [ x.tolower() if type(x) == str else  ]
    # print(len(collected_concepts), len(list_concepts))
    # print(len(set(collected_concepts)), len(set(list_concepts)))
    # print('speech synthesis' in collected_concepts)
    # for ss in list_concepts:
    #     print(ss)
    # print([ x for x in set(list_concepts) if x not in set(collected_concepts)])

    '''
        the  problem is that in the list_concepts, we have different concepts that represent the same
        meaning, 'Google DeepMind', 'Deepmind'.
        However, after the normalization, we found that there are less concepts due to duplication.

    '''