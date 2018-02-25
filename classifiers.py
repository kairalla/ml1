import sklearn
import sys
from sklearn.naive_bayes import MultinomialNB
from sklearn import svm
from sklearn import metrics

def main(argv):
	classifier = argv

	if classifier == 'MultinomialNB':
		clf = MultinomialNB()
	else:
		if classifier == 'SVM':
			clf = svm.SVC()
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

	f = open('data/out_classes_5.txt', 'r')
	lines = f.readlines()

	sentiments = [0] * len(lines)
	i = 0
	for line in lines:
		sentiments[i] = int(line)
		i += 1

	clf.fit(freq, sentiments)

	f = open('data/test_bag_of_words_0.csv', 'r')
	lines = f.readlines()

	test = [0]  * len(lines)
	i = 0
	for line in lines:
		counts = line.split(',')
		test[i] = [0] * len(counts)
		j = 0
		for val in counts:
			test[i][j] = int(val)
			j += 1
		i += 1

	predicted = clf.predict(test)

	f = open('data/test_classes_0.txt', 'r')
	lines = f.readlines()

	results = [0] * len(lines)
	i = 0
	for line in lines:
		results[i] = int(line)
		i += 1

	print metrics.accuracy_score(results , predicted)

if __name__ == "__main__":
  main(sys.argv[1])