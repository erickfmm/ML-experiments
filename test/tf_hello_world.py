import os
import sys
from os.path import dirname, join, abspath
sys.path.append(abspath(join(dirname(__file__), '..')))
######################################################

import tensorflow as tf
hello = tf.constant("hola mundo")
print(hello)
