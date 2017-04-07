from util import  *
import os
import unittest
import pprint
class TestParseMethods(unittest.TestCase):

    def test_parse_tag(self):
        import parser
        obj = parser.ParseTagEntity()
        self.assertTrue(len(obj.parse_tag())  > 0 )


    # def test_parse_entity(self):
    #     import parser
    #     obj = parser.ParseTagEntity()
    #     self.assertTrue(len(obj.parse_entity()) > 0)
    #
    # def test_plot_parse_tag(self):
    #     import parser
    #     obj = parser.ParseTagEntity()
    #     self.assertTrue(obj.plot_parse_tag())
    #
    # def test_plot_polarity(self):
    #     import parser
    #     obj = parser.ParseTagEntity()
    #     self.assertTrue(obj.plot_sparity())

#  class TestTagTagMethods(unittest.TestCase):
#
#     def test_cal_tag_tag_sim(self):
#         fs = os.listdir('query_pkl')[:5]
#         cal_tag_tag_sim(fs)
#
#     def test_plot_query(self,):
#         total = []
#         cuts = []
#         for i, fn in enumerate(os.listdir('query_pkl')):
#             #print(i)
#             tmp = get_query_result('query_pkl/' + fn)
#             total.extend(tmp)
#             gradient = np.gradient(sorted(tmp))
#             cutoff = len(np.where(gradient > 0.005)[0])
#             cuts.append(cutoff)
#             try :
#                 check_wierd_query('query_pkl/' + fn)
#             except:
#                 continue
#
#         indices = np.argsort(cuts)
#
#         plt.plot(cuts)
#         #plt.plot(sorted(total))
#         #plt.axvline(1000-cutoff)
#         #print(np.gradient(sorted(total)))
#
#         #print(np.gradient(np.gradient(sorted(total))))
#         plt.savefig('hist.png')
#         # plt.xlabel('1000 most similar words')
#         # plt.ylabel('Similarity')
#         # plt.savefig('sim_dynmaics_1000_examples.jpg')
#         # plt.close()
#         plt.close()
#
#     def test_plot_ind(self,):
#         plt.close()
#         fns = os.listdir('query_pkl')
#         f1 = fns[12]
#         f2 = fns[112]
#         #print(f1, f2)
#
#         d1 = pickle.load(open('query_pkl/' + f1, 'rb'))
#         d2 =  pickle.load(open('query_pkl/' + f2, 'rb'))
#         print(f1, d1)
#         print(f2, d2)

if __name__ == '__main__':
    unittest.main()