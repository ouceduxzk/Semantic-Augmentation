from sklearn.metrics.pairwise import cosine_similarity
import pickle
from gensim.corpora.mmcorpus import  MmCorpus
from gensim.corpora import wikicorpus
from gensim.models import TfidfModel
from gensim.similarities import Similarity, SparseMatrixSimilarity

def cosine_similarity(matrix):
    result = cosine_similarity(matrix, matrix)
    return result

def read_tfidfi_model():
    tfidf_fn = 'wiki_tfidf/_tfidf.mm'
    tfidf_mm = MmCorpus(tfidf_fn)
    #sim = Similarity('./', tfidf_mm, len(tfidf_mm))
    print(dir(tfidf_mm))
    index = SparseMatrixSimilarity(tfidf_mm)
    vec = [(1, 1), (4, 1)]
    print(tfidf_mm[vec])
    sims = index[tfidf_mm[vec]]
    print(list(enumerate(sims)))
    #pickle.dump(sim, open('tfidf_cosine.pkl', 'wb'))

def extract_title():
    data = wikicorpus.extract_pages('enwiki-latest-pages-articles.xml.bz2')



read_tfidfi_model()