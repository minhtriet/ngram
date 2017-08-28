import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import numpy as np
import pdb

if len(sys.argv) < 3:
    print "python compare.py predict_file ground_truth_file"
    sys.exit(1)

with open( sys.argv[1], 'r') as f:
    y_pred = f.read().splitlines()

with open( sys.argv[2], 'r') as f:
    y_true = f.read().splitlines()

y_pred = np.array(map(int, y_pred))
y_true = np.array(map(int, y_true))
cm = confusion_matrix(y_true, y_pred)
plt.figure()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.show()

accuracy = (y_pred == y_true).mean()

print "accuracy %f" % accuracy
