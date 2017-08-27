import sys
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

f = open(sys.argv[1], 'r')
y_pred = f.read().split('\n')
f.close()

f = open(sys.argv[2], 'r')
y_true = f.read().split('\n')
f.close()

cm = confusion_matrix(y_true, y_pred)
plt.figure()
plt.ylabel('True label')
plt.xlabel('Predicted label')

plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.show()
