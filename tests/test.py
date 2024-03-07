import torch

from test3 import *
import random

from collections import OrderedDict


seq = ['a', 'b', 'a', 'd', 'a', 'f', 'g', 'h', 'i', 'j']

seen = set()


unique_strings =  list(OrderedDict.fromkeys(seq))

print(unique_strings)




get_seed()