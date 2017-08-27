import sys
from sklearn.metrics import confusion_matrix

f = open(sys.argv[1], 'r') 
y_pred = f.read().split('\n')
f.close()

f = open(sys.argv[2], 'r') 
y_true = f.read().split('\n')
f.close()

confusion_matrix(y_true, y_pred)

