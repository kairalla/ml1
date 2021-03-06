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
from sklearn.tree import DecisionTreeClassifier
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
def main(argv):
    kBestFactor = .5
	# Choose classifier
    classifier = argv
    tfidf = TfidfTransformer(norm="l2", use_idf=True, smooth_idf=True, sublinear_tf=False)
    if classifier == 'MultinomialNB':
        clf = MultinomialNB()
    elif classifier == 'SVM':
        clf = svm.SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
    decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
    max_iter=-1, probability=True, random_state=None, shrinking=True,
    tol=0.001, verbose=False)
    elif classifier == "KNN":
        clf = KNeighborsClassifier(n_neighbors=13)
    elif classifier == "RF":
        clf = RandomForestClassifier(n_estimators=105)
    elif classifier == "DT":
        clf = DecisionTreeClassifier()
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
    # Fit the data
    chi = SelectKBest(chi2, k=int(len(freq[0])*kBestFactor))
    freq2 = chi.fit_transform(freq, sentiments)
    support = chi.get_support()
    # print support
    clf.fit(freq2, sentiments)

    # Read in test bag of words, tfidf transform, and predict
    f = open('data/test_bag_of_words_0.csv', 'r')
    lines = f.readlines()

    test = [0] * len(lines)
    i = 0
    for line in lines:
        counts = line.split(',')
        test[i] = [0] * int(len(counts)* kBestFactor)
        j = 0
        sup = 0
        for val in counts:
            if support[sup]:
                test[i][j] = int(val)
                j += 1
            sup += 1
        i += 1

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
    predictedProb = clf.predict_proba(test)
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    fpr, tpr, _ = roc_curve(results, predictedProb[:, 1])
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
