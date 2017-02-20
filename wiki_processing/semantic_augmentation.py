import os, sys , pickle
import numpy as np



def semantic_aug1(Tag_tag_matrix, Tag_obj_matrix):
    for col in range(Tag_obj_matrix.shape[1]):
        tmp = Tag_obj_matrix[:,col]
        nonzs_indices = np.nonzero(tmp)
        for ind in nonzs_indices:
            sim = getMostSim(tmp[ind]):
