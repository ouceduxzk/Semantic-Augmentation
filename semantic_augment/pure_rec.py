import numpy  as np
import os, sys
import base_rec
from sklearn.metrics.pairwise import cosine_similarity

class PureEvaluation(base_rec.BaseEvaluation):
    def __init__(self, o_u_t, num_o, num_u):
        self.obj_user_tag_matrix = o_u_t
        self.num_obj = num_o
        self.num_user = num_u
        self.sim_user_dict = {}
        self.sim_obj_user = cosine_similarity(self.obj_user_tag_matrix[:self.num_obj,:], \
                                         self.obj_user_tag_matrix[self.num_obj:,:])
        print('row : obj and users')
        print(self.sim_obj_user.shape)

    def sim(self, ind):
        assert ind < self.sim_obj_user.shape[1] and ind >= 0, 'index is out of bound'
        unsorted_sim_obj = self.sim_obj_user[:, ind]
        return unsorted_sim_obj


if __name__ == '__main__':
    dum = np.random.random((5,4))
    print(dum)
    test = PureEvaluation(dum, 2, 3)
    u = 0
    num_o = 2
    print('the query user index is {}'.format(u))
    print test.sim_obj_user
    print(test.sim(u))
