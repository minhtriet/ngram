import pickle
import pdb
import numpy as np

num_classes = 3
ngrams = [5, 6] #5-7

with open('xval.txt', 'r') as f:
    x = f.read().splitlines()

ngram_list = [ [0] * num_classes ] * len(ngrams) 

print('Loading ngram models')
for i in range(num_classes):
    for n in range(len(ngrams)):
        with open("class_%d_%d_gram" % ( i, ngrams[ n ] ), 'rb') as f:
            ngram_list[n][i] = pickle.load(f)
print('Loading completed')
print('Start classifying')

with open('result.txt', 'w') as f:
    for n_index, n in enumerate(ngrams):
        for string in x:
            classification = np.ones((len(ngrams), num_classes))
            for i in range(len(string)-n-1):
                for c in range(num_classes):
                    temp_string = string[i:i+n-1]
                    if temp_string in ngram_list[ n_index ][ c ]:
                        classification[n_index][ c ] = ngram_list[ n_index ][ c ][ temp_string ][ ord(string[i+n]) - 97 ] * classification[n_index][ c ]
                sums = np.sum(classification, axis=1)
                classification = classification / sums

            classification = np.array(classification)
            classification = np.sum(classification, axis=0)
            f.write( "%s\n" % np.argmax(classification, axis = 0) )
