import os
import sys
import datetime
from collections import Counter
import pickle

sys.path.remove('/home1/c/cis530/Software/python2.6/site-packages/rpy2-2.3.7-py2.7-linux-x86_64.egg')
sys.path.remove('/home1/c/cis530/Software/python2.6/site-packages')

reload(sys)
sys.setdefaultencoding('utf8')

print '==================='
print 'start job: counting word freq along time series'
start_time = datetime.datetime.now()
print 'start time: ' + str(start_time.ctime())
print '==================='


def get_all_token(dict_dir):
    unique_word = set()
    unique_date = set()
    for file_name in os.listdir(dict_dir):
        if file_name.startswith('1'):        
            unique_date.add(file_name)
            with open(dict_dir + '/' + file_name, 'r') as f:
                for file_line in f:
                    for word in file_line.split():
                        unique_word.add(word)
            f.close()
    return unique_word, unique_date


def initialize_dict(dict_dir):
    unique_word = set()
    unique_date = set()
    for file_name in os.listdir(dict_dir):
        if file_name.startswith('2'):
            unique_date.add(file_name)
            with open(dict_dir + '/' + file_name, 'r') as f:
                for file_line in f:
                    for word in file_line.split():
                        unique_word.add(word)
            f.close()
    result_map = {}
    for word in unique_word:
        result_map[word] = {}
        for date in unique_date:
            result_map[word][date] = 0
        if len(result_map) % 500000 == 0:
            logging(str(len(result_map)))
    return result_map


def logging(file_name):
    with open('/nlp/users/mazhiyu/word_time_series/gto_log/gto.log', 'a') as log_file:
        log_file.write(file_name + ' completes at: ' + str(datetime.datetime.now()) + '\n')
    log_file.close()


def go_thru_file_in_dir(category_timestamp_dir, empty_dict):
    for file_name in os.listdir(category_timestamp_dir):
        with open(category_timestamp_dir + '/' + file_name, 'r') as f:
            current_freq = {}
            for file_line in f:
                current_freq = dict(Counter(file_line.split()))
            for token in current_freq.keys():
                empty_dict[token][file_name] = current_freq[token]
        f.close()
        logging(file_name)
    with open('/nlp/users/mazhiyu/word_time_series/saved_variable/gto.pickle', 'w') as handler:
        pickle.dump(empty_dict, handler, protocol=pickle.HIGHEST_PROTOCOL)
    return empty_dict


#word_dim, date_dim = get_all_token('/nlp/users/mazhiyu/word_time_series/output')
#with open('/nlp/users/mazhiyu/word_time_series/saved_variable/simp_gto_word_dim.pickle', 'w') as handler:
#    pickle.dump(word_dim, handler, protocol=pickle.HIGHEST_PROTOCOL)
#with open('/nlp/users/mazhiyu/word_time_series/saved_variable/simp_gto_date_dim.pickle', 'w') as handler:
#    pickle.dump(date_dim, handler, protocol=pickle.HIGHEST_PROTOCOL)

#logging('word_dim, date_dim')

#word_dim = set()
#date_dim = set()

#with open('/nlp/users/mazhiyu/word_time_series/saved_variable/gto_word_dim.pickle', 'rb') as handler:
#    word_dim = pickle.load(handler)

#with open('/nlp/users/mazhiyu/word_time_series/saved_variable/gto_date_dim.pickle', 'rb') as handler:
#    date_dim = pickle.load(handler)

#logging('finish loading pickles...')
#logging('word size: ' + str(len(word_dim)))

#initial_dict = initialize_dict(word_dim, list(range(1, 7000)))
initial_dict = initialize_dict('/nlp/users/mazhiyu/word_time_series/output')
#with open('/nlp/users/mazhiyu/word_time_series/saved_variable/gto_initial_dict.pickle', 'w') as handler:
#    pickle.dump(initial_dict, handler, protocol=pickle.HIGHEST_PROTOCOL)
logging('initialization dict')
#result_dict = go_thru_file_in_dir('/nlp/users/mazhiyu/word_time_series/output', initial_dict)

print '==================='
print 'end job: counting word freq along time series'
end_time = datetime.datetime.now()
print 'end time: ' + str(end_time.ctime())
print 'job duration: ' + str(end_time - start_time)
print '==================='

