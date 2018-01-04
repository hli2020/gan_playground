import numpy as np
import random
import math
from scipy.misc import imread, imresize


def merge_dicts(src, dest):
    for key in src:
        if key not in dest:
            dest[key] = src[key]
        else:
            if type(dest[key]) is not type(src[key]):
                raise Exception("Unsupported")

            elif type(dest[key]) is list:
                dest[key].extend(src[key])
            elif type(dest[key]) is dict:
                dest[key] = merge_dicts(src[key], dest[key])
            elif type(dest[key]) is set:
                dest[key] = src[key] | dest[key]
            else:
                dest[key] = src[key]
    return dest

src = {
    'a': 4,
    'b': [5, 6, 7],
    'c': {
        'd': 'funny',
        'e': 5,
        'f': [1, 2, 3],
    },
    'new': {-1, -2, -3},
}

dest = {
    'a': 1000,
    'b': [5, 89, 70],
    'c': {
        'x': 'clarifai',
        'y': 8,
        'z': [10, 20, 30],
    },
    'new': {4, 5, 6, -1},
}


result = merge_dicts(src, dest)


