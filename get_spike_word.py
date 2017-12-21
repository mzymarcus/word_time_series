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
    with open('/nlp/users/mazhiyu/word_time_series/gto_log/spike_token_new.log', 'a') as log_file:
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

def category_date_by_month(date_dict):
    month_dict_list = []
    for year in range(1994, 2011):
        for month in range(1, 13):
            month = format(month, '02d')
            dates = []
            for date_key in date_dict.keys():
                if(date_key.startswith(str(year)+str(month))):
                    dates.append(date_dict[date_key])
            if dates:
                month_dict = {}
                month_dict[str(year) + str(month)] = dates
                month_dict_list.append(month_dict)
    return month_dict_list

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
'''
with open('/nlp/users/mazhiyu/word_time_series/saved_variable/dict_gto_date_dim.pickle', 'rb') as handler:
    date_dict = pickle.load(handler)

logging('finish loading pickles...')
logging('word size: ' + str(len(word_dict)))
logging('date size: ' + str(len(date_dict)))

loader = numpy.load('saved_variable/result_csr_matrix.npz')
csr_matrix = csr_matrix((loader['data'], loader['indices'], loader['indptr']), shape=loader['shape'], dtype=numpy.float32)

#csr_matrix.data
#csr_matrix.todense()
matrix_2d = csr_matrix.toarray().reshape((8544617, 6077))
matrix_2d_normalized = normalize(matrix_2d, axis=1)

month_dict_list = category_date_by_month(date_dict)
spike_token = {}

for i in range(8544617):
    logging(str(i))
    for month_dict in month_dict_list:
        month = month_dict.keys()[0]
        dates_index = month_dict[month]
        token = matrix_2d[i]
        # token = numpy.ma.array(token, mask=True)
        # for date in dates_index:
        #     token.mask[date] = False
        if token[dates_index].sum() >= 1000:
            spike_token[i] = month
            break

with open('/nlp/users/mazhiyu/word_time_series/saved_variable/spike_token_new.pickle', 'w') as handler:
    pickle.dump(spike_token, handler, protocol=pickle.HIGHEST_PROTOCOL)

with open('/nlp/users/mazhiyu/word_time_series/saved_variable/spike_token_new.pickle', 'rb') as handler:
    spike_token = pickle.load(handler)

with open('gto_log/spike_token_new.txt', 'w') as filehandler:
    for key in spike_token.keys():
        filehandler.write(str(key) + ', ' + spike_token[key] + '\n')
filehandler.close()
'''
with open('/nlp/users/mazhiyu/word_time_series/saved_variable/spike_token_new.pickle', 'rb') as handler:
    token_dict = pickle.load(handler)

with open ('gto_log/spike_token_new_with_word.txt', 'w') as file1:
    for key in token_dict.keys():
        value = token_dict[key]    
        word = word_dict.keys()[word_dict.values().index(key)]
        file1.write(word + ', ' + str(key) + ', ' + value + '\n')
file1.close()

print '==================='
print 'end job: counting word freq along time series'
end_time = datetime.datetime.now()
print 'end time: ' + str(end_time.ctime())
print 'job duration: ' + str(end_time - start_time)
print '==================='

