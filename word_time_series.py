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

import datetime
import os
import gzip
import re
from nltk.tokenize import word_tokenize
import pickle

print '==================='
print 'start job: counting word freq along time series'
print 'start time: ' + str(datetime.datetime.now().ctime())
print '==================='

current_timestamp = 'null_date'
current_freq = {}
token_freq = {}

for file_name in os.listdir('/nlp/data/corpora/LDC/LDC2003T05/'):
    if not file_name.startswith('.'):
        with gzip.open('/nlp/data/corpora/LDC/LDC2003T05/' + file_name, 'r') as handler:
            for file_line in handler:
                if file_line.startswith('<'):
                    if file_line.startswith('<DOC'):
                        if current_timestamp != 'null_date':
                            token_freq[current_timestamp] = current_freq
                            current_freq = {}
                        indices = [index.start() for index in re.finditer('"', file_line)]
                        current_timestamp = file_line[(indices[0]+1):(indices[1])]
                else:
                    if current_timestamp != 'null_date':
                        word_list = word_tokenize(file_line)
                        for word in word_list:
                            if word in current_freq:
                                current_freq[word] = current_freq[word] + 1
                            else:
                                current_freq[word] = 1

with open('sorted_freq', 'wb') as handler:
    pickle.dump(token_freq, handler, protocol=pickle.HIGHEST_PROTOCOL)

print '==================='
print 'end job: counting word freq along time series'
print 'end time: ' + str(datetime.datetime.now().ctime())
print '==================='

