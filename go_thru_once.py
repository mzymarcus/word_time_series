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
import faiss
from sklearn.preprocessing import normalize

print '==================='
print 'start job: counting word freq along time series'
start_time = datetime.datetime.now()
print 'start time: ' + str(start_time.ctime())
print '==================='


def get_all_token(dict_dir):
    unique_word = set()
    unique_date = set()
    for file_name in os.listdir(dict_dir):
        #if file_name.startswith('1'):        
        unique_date.add(file_name)
        with open(dict_dir + '/' + file_name, 'r') as f:
            for file_line in f:
                for word in file_line.split():
                    unique_word.add(word)
        f.close()
    unique_word_dict = {key: value for value, key in enumerate(list(unique_word))} 
    unique_date_dict = {key: value for value, key in enumerate(list(unique_date))}
    return unique_word_dict, unique_date_dict


def initialize_dict(word_dimension, date_dimension):
    result_map = {}
    for word in word_dimension:
        result_map[word] = {}
        for date in date_dimension:
            result_map[word][date] = 0
        if len(result_map) % 500000 == 0:
            logging(str(len(result_map)))
    return result_map


def initialize_numpy_matrix(word_dimension, date_dimension):
    return numpy.zeros((len(word_dimension.keys()), len(date_dimension.keys())), dtype=int16)

    
def logging(file_name):
    # with open('/nlp/users/mazhiyu/word_time_series/gto_log/gto.log', 'a') as log_file:
    with open('/nlp/users/mazhiyu/word_time_series/gto_log/knn_cos.log', 'a') as log_file:
        log_file.write(file_name + ' completes at: ' + str(datetime.datetime.now()) + '\n')
    log_file.close()


def go_thru_file_in_dir(category_timestamp_dir, empty_dict, word_dict, date_dict):
    for file_name in os.listdir(category_timestamp_dir):
        logging('begin processing: ' + file_name)
        with open(category_timestamp_dir + '/' + file_name, 'r') as f:
            current_freq = {}
            for file_line in f:
                current_freq = dict(Counter(file_line.split()))
            for token in current_freq.keys():
                empty_dict[word_dict[token]][date_dict[file_name]] = current_freq[token]
        f.close()
        logging('finish processing: ' + file_name)
    #with open('/nlp/users/mazhiyu/word_time_series/saved_variable/gto.pickle', 'w') as handler:
    #    pickle.dump(empty_dict, handler, protocol=pickle.HIGHEST_PROTOCOL)
    return empty_dict


#word_dim, date_dim = get_all_token('/nlp/users/mazhiyu/word_time_series/output')
#with open('/nlp/users/mazhiyu/word_time_series/saved_variable/dict_gto_word_dim.pickle', 'w') as handler:
#    pickle.dump(word_dim, handler, protocol=pickle.HIGHEST_PROTOCOL)
#with open('/nlp/users/mazhiyu/word_time_series/saved_variable/dict_gto_date_dim.pickle', 'w') as handler:
#    pickle.dump(date_dim, handler, protocol=pickle.HIGHEST_PROTOCOL)

#logging('word_dim, date_dim')


word_dict = {}
date_dict = {}

with open('/nlp/users/mazhiyu/word_time_series/saved_variable/dict_gto_word_dim.pickle', 'rb') as handler:
    word_dict = pickle.load(handler)

with open('/nlp/users/mazhiyu/word_time_series/saved_variable/dict_gto_date_dim.pickle', 'rb') as handler:
    date_dict = pickle.load(handler)

logging('finish loading pickles...')
logging('word size: ' + str(len(word_dict)))
logging('date size: ' + str(len(date_dict)))

'''
#initial_dict = initialize_dict(word_dim, list(range(1, 7000)))
initial_dict = initialize_numpy_matrix(word_dim, date_dim)

#with open('/nlp/users/mazhiyu/word_time_series/saved_variable/gto_initial_dict.pickle', 'w') as handler:
#    pickle.dump(initial_dict, handler, protocol=pickle.HIGHEST_PROTOCOL)
logging('initialization dict')

result_dict = go_thru_file_in_dir('/nlp/users/mazhiyu/word_time_series/output', initial_dict, word_dim, date_dim)

result_csr_matrix = sparse.csr_matrix(result_dict)

numpy.savez('saved_variable/result_csr_matrix', data=result_csr_matrix.data, indices=result_csr_matrix.indices, indptr=result_csr_matrix.indptr, shape=result_csr_matrix.shape)
'''

loader = numpy.load('../saved_variable/result_csr_matrix.npz')
csr_matrix = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'], dtype=numpy.float32)

#csr_matrix.data
#csr_matrix.todense()
matrix_2d = csr_matrix.toarray().reshape((8544617, 6077))
matrix_2d = normalize(matrix_2d, axis=1)

index = faiss.IndexFlatIP(6077)
index.add(matrix_2d)

test_token_index = [7311612, 697832, 911930, 2084024, 8021496, 6142395, 3207073, 2339412, 4372283, 8519331]
test_token = ['obama', 'spring', 'summer', 'fall', 'winter', 'football', 'election', 'midterm', 'volcano', 'oscar']

for iteration in range(len(test_token_index)):
    query = matrix_2d[test_token_index[iteration]].reshape((1, 6077))
    distance, indices = index.search(query, 101)
    logging('indices for nearest neighbors for ' + test_token[iteration] + ':')
    logging('    ' + str(indices[0]))
    logging('distance for nearest neighbors for ' + test_token[iteration] + ':')
    logging('    ' + str(distance[0]))
    logging('token for nearest neighbors for ' + test_token[iteration] + ':')
    knn_neighbor = []
    for tindex in indices[0]:
        knn_neighbor.append(word_dict.keys()[word_dict.values().index(tindex)])
    logging('    ' + str(knn_neighbor))

print '==================='
print 'end job: counting word freq along time series'
end_time = datetime.datetime.now()
print 'end time: ' + str(end_time.ctime())
print 'job duration: ' + str(end_time - start_time)
print '==================='

