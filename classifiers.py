import numpy as np
import matplotlib.pyplot as plt
import sklearn
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import roc_curve, auc


def main(argv):
	# Choose classifier
    classifier = argv
    tfidf = TfidfTransformer(norm="l2",use_idf=True, smooth_idf=True, sublinear_tf=False)
    if classifier == 'MultinomialNB':
        clf = MultinomialNB()
    elif classifier == 'SVM':
        clf = svm.SVC()
    elif classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=12)
    elif classifier == "RandomForest":
        clf = RandomForestClassifier(n_estimators=100)
    else:
        print "No such classifier"
        return

    # Read in training bag of words and tfidf transform
    f = open('data/out_bag_of_words_5.csv', 'r')
    lines = f.readlines()

    freq = [0] * len(lines)
    i = 0
    for line in lines:
        counts = line.split(',')
        freq[i] = [0] * len(counts)
        j = 0
        for val in counts:
            freq[i][j] = int(val)
            j += 1
        i += 1

    tfidf.fit_transform(freq, y=None)

    # Read in classes
    f = open('data/out_classes_5.txt', 'r')
    lines = f.readlines()

    sentiments = [0] * len(lines)
    i = 0
    for line in lines:
        sentiments[i] = int(line)
        i += 1

    # tfidf.fit(freq, sentiments)

    # Fit the data
    clf.fit(freq, sentiments)

    # Read in test bag of words, tfidf transform, and predict
    f = open('data/test_bag_of_words_0.csv', 'r')
    lines = f.readlines()

    test = [0] * len(lines)
    i = 0
    for line in lines:
        counts = line.split(',')
        test[i] = [0] * len(counts)
        j = 0
        for val in counts:
            test[i][j] = int(val)
            j += 1
        i += 1

    tfidf.transform(test)
    predicted = clf.predict(test)

    # Read in test classes and measure accuracy
    f = open('data/test_classes_0.txt', 'r')
    lines = f.readlines()

    results = [0] * len(lines)
    i = 0
    for line in lines:
        results[i] = int(line)
        i += 1

    print metrics.accuracy_score(results, predicted)

    # Calculate ROC curve
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(results, predicted)
    roc_auc = auc(fpr, tpr)

    # Plot ROC
    plt.figure()
    lw = 1
    plt.plot(fpr, tpr, color='darkorange', lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(argv + ' ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

if __name__ == "__main__":
    main(sys.argv[1])
