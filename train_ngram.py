import pdb
import numpy as np
import pickle

num_classes = 12

with open('xtrain_obfuscated.txt', 'r') as f:
    x = f.read().splitlines()
    
with open('ytrain.txt', 'r') as f:
    y = f.read().splitlines()

dataset = [[] for i in range(num_classes)] 

for index, value in enumerate(x):
    dataset[ int(y[index]) ].append(value)

ngram = [1,2,3,4]
for n in ngram:
    for index, c in enumerate(dataset):
        print 'Computing %d_gram for class %d' % (n,index)
        grams = {} 
        for string in c:
            for i in xrange(len(string)-n-1):
                key = string[i:i+n-1]
                value = ord(string[i+n]) - 97
                if key in grams:
                    grams[key][value] = grams[key][value] + 1
                else:
                    grams[key] = [0]*26
                    grams[key][value] = 1
        # frequency
        total = np.sum( np.array(grams.values()) ) * 1.0
        new_values = np.array(grams.values()) / total
        grams = dict(zip(grams.keys(), new_values))
        with open("class_%s_%d_gram" % (index, n), 'wb') as f:
            pickle.dump(grams, f)
        print 'Result is written to class_%s_%d_gram' % (index, n)
