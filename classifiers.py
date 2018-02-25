import sklearn
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn import svm
from sklearn import metrics


def main(argv):
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
    print "whew"
    f = open('data/out_classes_5.txt', 'r')
    lines = f.readlines()

    sentiments = [0] * len(lines)
    i = 0
    for line in lines:
        sentiments[i] = int(line)
        i += 1
    # tfidf.fit(freq, sentiments)
    clf.fit(freq, sentiments)

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

    f = open('data/test_classes_0.txt', 'r')
    lines = f.readlines()

    results = [0] * len(lines)
    i = 0
    for line in lines:
        results[i] = int(line)
        i += 1

    print metrics.accuracy_score(results, predicted)


if __name__ == "__main__":
    main(sys.argv[1])
