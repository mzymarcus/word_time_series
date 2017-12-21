import os
import sys
import datetime
from collections import Counter
import pickle

sys.path.remove('/home1/c/cis530/Software/python2.6/site-packages/rpy2-2.3.7-py2.7-linux-x86_64.egg')
sys.path.remove('/home1/c/cis530/Software/python2.6/site-packages')

reload(sys)
sys.setdefaultencoding('utf8')

import numpy
test_matrix_float = numpy.ones((1000, 7000))

numpy.savetxt('float_matrix.txt', test_matrix_float)

test_matrix_int = test_matrix_float.astype(int)

numpy.savetxt('int_matrix.txt', test_matrix_int)

print 'success'
