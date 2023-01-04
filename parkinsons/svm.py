import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import svm
import numpy as np
from sklearn import preprocessing
from sklearn.metrics import average_precision_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import plot_precision_recall_curve
from sklearn.metrics import accuracy_score
from sklearn.metrics import plot_confusion_matrix
from sklearn import metrics
import matplotlib.pyplot as plt

data = pd.read_csv("C:\\Users\\Jessica\\Dropbox\\My PC (DESKTOP-L1KFBMS)\\Downloads\\parkinsons - parkinsons.csv")
target = pd.read_csv("C:\\Users\\Jessica\\Dropbox\\My PC (DESKTOP-L1KFBMS)\\Downloads\\Parkinson's_Target - parkinsons (1).csv")
data = preprocessing.normalize(data, norm='l2', axis=0, copy=True, return_norm=False)
data_train, data_test, target_train, target_test = train_test_split(data, target, test_size=.25)

random_state = np.random.RandomState(0)
classifier = svm.SVC(random_state=random_state)
classifier.fit(data_train, np.ravel(target_train))
target_score = classifier.decision_function(data_test)
average_precision = average_precision_score(target_test, target_score)
print('Average precision-recall score: {0:0.2f}'.format(average_precision))
np.set_printoptions(precision=2)
disp = plot_precision_recall_curve(classifier, data_test, target_test)
disp.ax_.set_title('Precision-Recall curve: ''AP={0:0.2f}'.format(average_precision))
predicted = classifier.predict(data_test)
actual = target_test
print(accuracy_score(actual, predicted))
fpr, tpr, thresholds = metrics.roc_curve(target_test, target_score, pos_label=1)
roc_auc = metrics.auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend(loc="lower right")
plt.show()
title = ("Normalized confusion matrix")
disp2 = plot_confusion_matrix(classifier, data_test, target_test, display_labels=[0,1], cmap=plt.cm.Blues)
disp2.ax_.set_title(title)
print(title)
print(disp2.confusion_matrix)
plt.show()
