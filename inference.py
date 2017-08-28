import pickle
import sys
import numpy as np

if len(sys.argv) < 3:
    print "python inference.py input_file output_file"
    sys.exit(1)

num_classes = 12
ngrams = [4,5,6,7]

def normalize(x):
    if len(np.unique(x)) == 1:
        x = np.ones(len(x))
    pos_min = np.min(x[x > 0])
    x[x == 0] = pos_min / 2  # filter value
    sums = np.sum(x)    
    return x/sums

with open( sys.argv[1], 'r') as f:
    x = f.read().splitlines()

ngram_list = np.empty((len(ngrams), num_classes), dtype=object)

print('Loading ngram models')
for i in range(num_classes):
    for n in range(len(ngrams)):
        with open("class_%d_%d_gram" % ( i, ngrams[ n ] ), 'rb') as f:
            ngram_list[n][i] = pickle.load(f)

print('Loading completed')
print('Start classifying')
count = 0

with open( sys.argv[2], 'w') as f:
    for string in x:
        count = count + 1
        if count % 20 == 0:
            print "classifying sentence %d " % count
        classification = np.ones((len(ngrams), num_classes))
        for n_index, n in enumerate(ngrams):
            for i in range(len(string)-n-1): # interate through each substring
                temp_string = string[i:i+n-1]
                temp_string_classification = np.zeros(num_classes)
                for c in range(num_classes): # classify the patch
                    if temp_string in ngram_list[ n_index ][ c ]:
                        temp_string_classification[ c ] = ngram_list[ n_index ][ c ][ temp_string ][ ord(string[i+n]) - 97 ]
                temp_string_classification = normalize(temp_string_classification) # make sure each ngram sums up to 1
                classification[n_index] = np.multiply(classification[n_index], temp_string_classification)
                sums = np.sum(classification[n_index])
                classification[n_index] = np.divide(classification[n_index], sums)

        classification = np.sum(classification, axis=0)
        f.write( "%s\n" % np.argmax(classification, axis = 0) )
