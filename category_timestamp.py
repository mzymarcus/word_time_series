#!/usr/bin/python
#$ -S /usr/bin/python
# other OGS/SGE options go here, each line beginning with #$
#
# to add modules for your own research, use pip inside a virtualenv. see
# these instructions (skip the first step; virtualenv itself is already installed)
# http://docs.python-guide.org/en/latest/dev/virtualenvs/
#
# inside that virtual environment, you can pip install your desired modules
# and maintain the exact version of each that you might need
#
# make sure to replace ${YOUR_VIRTUAL_ENV_DIR} below with the
# actual path to wherever you created your virtualenv above.

# source ${YOUR_VIRTUAL_ENV_DIR}/bin/activate


import os
import gzip
import re
from nltk.tokenize import word_tokenize
import sys
import datetime
import fcntl
from collections import Counter
import pickle
from subprocess import call
import subprocess

sys.path.remove('/home1/c/cis530/Software/python2.6/site-packages/rpy2-2.3.7-py2.7-linux-x86_64.egg')
sys.path.remove('/home1/c/cis530/Software/python2.6/site-packages')

reload(sys)
sys.setdefaultencoding('utf8')


def calculate_token_freq_from_corpus(corpus_dir):
    print 'reading corpus for: ' + corpus_dir
    token_freq = []
    for file_name in os.listdir(corpus_dir):
        if file_name.endswith('.gz'):
            print '    start processing file: ' + file_name
            current_timestamp = 'null_date'
            with gzip.open(corpus_dir + file_name if corpus_dir.endswith('/')
                           else corpus_dir + '/' + file_name, 'r') as handler:
                for file_line in handler:
                    if file_line.startswith('<'):
                        if file_line.startswith('<DOC'):
                            tmp_timestamp = file_line[re.search('\d', file_line).start():file_line.find('.')]
                            if current_timestamp != 'null_date' and current_timestamp != tmp_timestamp:
                                with open('/nlp/users/mazhiyu/word_time_series/output_new/' + current_timestamp, 'a') as f:
                                    #fcntl.flock(f, fcntl.LOCK_EX)
                                    f.write(' '.join(token_freq))
                                    #fcntl.flock(f, fcntl.LOCK_UN)
                                f.close()
                                token_freq = []
                            current_timestamp = tmp_timestamp
                    else:
                        try:
                            # word_list = word_tokenize(file_line)
                            word_list_tmp = subprocess.Popen(['./segment.sh', 'ctb', file_line, 'UTF-8', '0'], stdout=subprocess.PIPE)
                            word_list = word_list_tmp.stdout.read().split(' ')
                            print('word_list: ' + word_list)
                            token_freq = token_freq + word_list
                        except:
                            print 'encoding problem with line: ' + file_line + ' on timestamp: ' + current_timestamp
                try:
                    # word_list = word_tokenize(file_line)
                    word_list_tmp = subprocess.Popen(['./segment.sh', 'ctb', file_line, 'UTF-8', '0'], stdout=subprocess.PIPE)
                    word_list = word_list_tmp.stdout.read().split(' ')
                    print('word_list: ' + word_list)
                    token_freq = token_freq + word_list
                except:
                    print 'encoding problem with line: ' + file_line + ' on timestamp: ' + current_timestamp
                finally:
                    with open('/nlp/users/mazhiyu/word_time_series/output_new/' + current_timestamp, 'a') as f:
                        f.write(' '.join(token_freq))
                    f.close()
                    token_freq = []


def get_final_freq(category_timestamp_dir):
    #all_timestamp = set()
    final_result = {}
    #for file_name in os.listdir(category_timestamp_dir):
    #    all_timestamp.add(file_name)
    counter = 0
    for file_name in os.listdir(category_timestamp_dir):
        counter = counter + 1
        with open(category_timestamp_dir + '/' + file_name, 'r') as f:
            current_freq = {}
            for file_line in f:
                current_freq = dict(Counter(file_line.split()))
            for token in current_freq.keys():
                if token in final_result.keys():
                    final_result[token][file_name] = current_freq[token]
                else:
                    final_result[token] = {}
                    #for date in all_timestamp:
                    #    final_result[token][date] = 0
                    final_result[token][file_name] = current_freq[token]
        f.close()
        if counter % 20 == 1:
            with open('/nlp/users/mazhiyu/word_time_series/log/combined_dict_job_output', 'w') as f:
                f.write(str(counter) + ' ' + str(datetime.datetime.now().ctime()) + '\n')
            f.close()
            print_dict(final_result)
            with open('/nlp/users/mazhiyu/word_time_series/log/word_freq_pickle', 'w') as handler:
                pickle.dump(final_result, handler, protocol=pickle.HIGHEST_PROTOCOL)
    return final_result


def print_dict(dict_to_print):
    f = open('/nlp/users/mazhiyu/word_time_series/log/combined_dict_output', 'w')
    for key_2 in sorted(dict_to_print[dict_to_print.keys()[0]].keys()):
        f.write(key_2 + ',')
    f.write('\n')
    for key_1, value_1 in dict_to_print.items():
        f.write(key_1 + ',')
        for key_2 in sorted(value_1.keys()):
            f.write(str(value_1[key_2]) + ',')
        f.write('\n')
    f.close()


print '==================='
print 'start job: counting word freq along time series'
start_time = datetime.datetime.now()
print 'start time: ' + str(start_time.ctime())
print '==================='

corpus_dir_path = sys.argv[1]
calculate_token_freq_from_corpus(corpus_dir_path)

'''
final_dict = get_final_freq('/nlp/users/mazhiyu/word_time_series/output')
print_dict(final_dict)
with open('/nlp/users/mazhiyu/word_time_series/log/word_freq_pickle', 'w') as handler:
    pickle.dump(final_result, handler, protocol=pickle.HIGHEST_PROTOCOL)
'''
print '==================='
print 'end job: counting word freq along time series'
end_time = datetime.datetime.now()
print 'end time: ' + str(end_time.ctime())
print 'job duration: ' + str(end_time - start_time)
print '==================='
