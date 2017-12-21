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
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

print '==================='
print 'start job: counting word freq along time series'
start_time = datetime.datetime.now()
print 'start time: ' + str(start_time.ctime())
print '==================='


def logging(file_name):
    with open('/nlp/users/mazhiyu/word_time_series/gto_log/cos_sim.log', 'a') as log_file:
        log_file.write(file_name + ' completes at: ' + str(datetime.datetime.now()) + '\n')
    log_file.close()


numpy_dict = numpy.loadtxt('/nlp/users/mazhiyu/word_time_series/gto_log/result_dict_output.txt').reshape((8544617, 6077))

logging('finish loading numpy dict...')


