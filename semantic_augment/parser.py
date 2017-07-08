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
import codecs
class ParseTagEntity(object):
    def parse_tag(self):
        # standardize the concept 'scientific computing' to 'computational science'
        o2omap = pickle.load(open('standarize.pkl', 'rb'))
        lines = codecs.open('collected_data/pandodb_testing-tags','r', encoding='utf8' )
        objs = []
        users = []
        total = []
        for line in lines :
            data = json.loads(line)
            tag_name = data['name']
            isUser = True
            tag_id = data['_id']['$oid']
            wiki_url = data['concept_url'].split('wiki/')[-1].replace('_', ' ').lower()
            if wiki_url in correction.keys():
                wiki_url = correction[wiki_url]

            if wiki_url in o2omap.keys():
                wiki_url = o2omap[wiki_url]

            # if wiki_url.encode('utf-8').startswith('ha'):
            #     print(data['concept_url'], wiki_url)

            if u'sk infosec' == wiki_url:
                wiki_url = u'infosys'

            if u'tesla, inc.' == wiki_url:
                wiki_url =  u'tesla motors'

            if data['entity_url'].startswith('/public'):
                entity_id = unicode(data['entity_url'].split('/')[-1])
                users.append([ entity_id, wiki_url])
                total.append([entity_id, wiki_url])
            else:
                entity_id = unicode(data['entity_url'])
                objs.append([entity_id, wiki_url])
                total.append([entity_id, wiki_url])

        total[427][1] = 'ille'
        user_entities = [x[0] for x in users]
        user_entities = [k for k, v in Counter(user_entities).iteritems() if v > 4]
        users     = [ [e, t] for e, t in users if e in user_entities]

        #users = [[a, o2omap[b.decode('utf-8')] ] for a, b in users if a in user_entities]
        concepts = [ x[1] for x in users]
        tmp = Counter(concepts)
        ############################################################################
        # plt.plot(range(len(tmp)), sorted(tmp.values()) )
        # plt.ylabel('frequency')
        # plt.xlabel('index of each concept')
        # plt.title('user concept freq distribution')
        # plt.savefig('user_concept_freq.png')
        ############# resources are also part of objects ##########################
        part_obj = [[a, b] for a, b in users if a not in user_entities]
        objs = [ [a, b ] for a, b in objs] + part_obj
        #print('len of objs and users are {} and {}'.format(len(objs), len(users)))

        #total = [ [a, o2omap[b]] for a, b in total ]
        return objs, users, total

    def parse_obj_tag_matrix(self):
        objs, users, o_u = self.parse_tag()
        user_concepts = [ x[1] for x in users ]
        user_entities = [ x[0] for x in users ]

        concepts = list(set([x[1] for x in o_u]))
        ########################################################################
        #concepts = list(set(new_concepts))

        entities = list(set([x[0] for x in o_u]))
        print(entities)

        print('# of concepts {}'.format(len(concepts)))
        print('# of entities {}'.format(len(entities)))

        #@DONE arrange the entities such that the first 25 are users and the next 48 are objects
        tmp_objs =  list(set([ x for x in entities if x not in set(user_entities)]))
        print('len of objs and users are {} and {}'.format(len(tmp_objs), len(set(user_entities))))

        entities = list(set(user_entities)) + tmp_objs


        print('the number of entities are {}'.format(len(entities)))
        #@DONE build the obj/user and Tag matrix
        '''
            The local_tag_indexing is a map of all concepts to the local index
            The local_entity_indexing is a map of all entities to the local index while
            reverse_entity_indexing the the reverse map of local_index to the name of entity.
        '''

        ou_tag_indexing  =   dict([ (item,idx) for idx, item in enumerate(concepts) ])

        print(ou_tag_indexing)
        ou_entity_indexing = dict([ (item,idx) for idx, item in enumerate(entities) ])

        print(ou_entity_indexing)

        user_tag_indexing    = dict([(item, idx) for idx, item in enumerate(set(user_concepts))])
        user_entity_indexing = dict([(item, idx) for idx, item in enumerate(set(user_entities))])

        # reverse_tag_indexing = dict([(idx, item) for idx, item in enumerate(concepts)])
        # reverse_entity_indexing = dict([(idx, item) for idx, item in enumerate(entities)])

        user_row = []
        user_col = []
        ou_row = []
        ou_col = []

        for entity,concept  in o_u :
            ou_row.append(ou_entity_indexing[entity])
            ou_col.append(ou_tag_indexing[concept])

        for entity, concept in users :
            user_row.append(user_entity_indexing[entity])
            user_col.append(user_tag_indexing[concept])

        #ou_tag_matrix = sps.coo_matrix(([1] * len(objs + users), (ou_row, ou_col)), shape=(len(ou_entity_indexing), len(ou_tag_indexing))).tocsr()
        user_tag_matrix = sps.coo_matrix(([1] * len(users), (user_row, user_col)), shape=(len(user_entity_indexing), len(user_tag_indexing))).tocsc()
        #print('the shape of entity_tag matrix is row={}, col={}'.format(len(local_entity_indexing), len(local_tag_indexing)))
        #print(ou_tag_matrix.shape)
        entity_tag_matrix = sps.coo_matrix(([1] * len(o_u), (ou_row, ou_col)), shape=(len(ou_entity_indexing), len(ou_tag_indexing))).tocsr()
        print('shape of o_t matrix is')
        print(entity_tag_matrix.shape)

        if (not os.path.exists(('save'))):
            os.mkdir('save')
        print(user_tag_matrix.shape)
        pickle.dump(concepts, open('collected_concepts.pkl', 'wb'))
        pickle.dump(entities, open('collected_entities.pkl', 'wb'))
        fp = open('save/entity_tag_matrix.pkl', 'wb')
        pickle.dump(concepts, fp)
        pickle.dump(entity_tag_matrix, fp)
        pickle.dump(user_concepts, fp)
        pickle.dump(user_tag_matrix, fp)


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
    #obj.plot_parse_tag()
    obj.parse_obj_tag_matrix()

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