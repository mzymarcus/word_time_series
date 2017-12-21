import sys

sys.path.remove('/home1/c/cis530/Software/python2.6/site-packages/rpy2-2.3.7-py2.7-linux-x86_64.egg')
sys.path.remove('/home1/c/cis530/Software/python2.6/site-packages')

reload(sys)
sys.setdefaultencoding('utf8')

import numpy
import os
import datetime
from collections import Counter
import pickle
from scipy.sparse import *
from scipy import *
from scipy import sparse
from sklearn.preprocessing import normalize
from scipy import spatial

print '==================='
print 'start job: counting word freq along time series'
start_time = datetime.datetime.now()
print 'start time: ' + str(start_time.ctime())
print '==================='

    
def logging(file_name):
    with open('/nlp/users/mazhiyu/word_time_series/gto_log/simplex.log', 'a') as log_file:
        log_file.write(file_name + ' completes at: ' + str(datetime.datetime.now()) + '\n')
    log_file.close()

def load_simplex_score(filepath, word_dict):
    word_list = []
    simplex_score = {}
    with open(filepath) as filedes:
        for line in filedes:
            word_1 = line.split('\t')[0]
            word_2 = line.split('\t')[1]
            score = line.split('\t')[3]
            if word_1 not in word_dict.keys():
                logging('not found ' + word_1)
                continue
            if word_2 not in word_dict.keys():
                logging('not found ' + word_2)
                continue
            word_list.append(word_1 + '`' + word_2)
            simplex_score[word_1 + '`' + word_2] = float(score)
            logging('finish loading ' + word_1 + ' ' + word_2)
    return word_list, simplex_score


with open('/nlp/users/mazhiyu/word_time_series/saved_variable/dict_gto_word_dim.pickle', 'rb') as handler:
    word_dict = pickle.load(handler)

#word_list, simplex_score = load_simplex_score('SimLex-999.txt', word_dict)

#with open('/nlp/users/mazhiyu/word_time_series/saved_variable/word_list.pickle', 'w') as handler:
#    pickle.dump(word_list, handler, protocol=pickle.HIGHEST_PROTOCOL)

#with open('/nlp/users/mazhiyu/word_time_series/saved_variable/simplex_score.pickle', 'w') as handler:
#    pickle.dump(simplex_score, handler, protocol=pickle.HIGHEST_PROTOCOL)

with open('/nlp/users/mazhiyu/word_time_series/saved_variable/word_list.pickle', 'rb') as handler:
    word_list = pickle.load(handler)

with open('/nlp/users/mazhiyu/word_time_series/saved_variable/simplex_score.pickle', 'rb') as handler:
    simplex_score = pickle.load(handler)

loader = numpy.load('saved_variable/result_csr_matrix.npz')
csr_matrix = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'], dtype=numpy.float32)

#csr_matrix.data
#csr_matrix.todense()

matrix_2d = csr_matrix.toarray().reshape((8544617, 6077))
matrix_2d = normalize(matrix_2d, axis=1)

score = {}
cos_score = {}

for comb_word in word_list:
    word_1 = comb_word.split('`')[0]
    word_2 = comb_word.split('`')[1]
    index_1 = word_dict[word_1]
    index_2 = word_dict[word_2]
    score[word_1 + '    ' + word_2] = numpy.corrcoef(matrix_2d[index_1], matrix_2d[index_2])[0][1]
    cos_score[word_1 + '    ' + word_2] = 1 - spatial.distance.cosine(numpy.array(matrix_2d[index_1]), numpy.array(matrix_2d[index_2]))
    logging('finish computing ' + word_1 + '    ' + word_2)

with open('/nlp/users/mazhiyu/word_time_series/saved_variable/time_vector_coeff_score.pickle', 'w') as handler:
    pickle.dump(score, handler, protocol=pickle.HIGHEST_PROTOCOL)

with open('/nlp/users/mazhiyu/word_time_series/saved_variable/time_vector_cossim_score.pickle', 'w') as handler:
    pickle.dump(cos_score, handler, protocol=pickle.HIGHEST_PROTOCOL)    

print '==================='
print 'end job: counting word freq along time series'
end_time = datetime.datetime.now()
print 'end time: ' + str(end_time.ctime())
print 'job duration: ' + str(end_time - start_time)
print '==================='

