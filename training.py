from sklearn.neighbors import KNeighborsClassifier
from data_preparation import *
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier

data = pd.read_csv('Datasets/relevant_data/cleanedDataset.csv', index_col = 0)
test_data = pd.read_csv('Datasets/relevant_data/cleanedTestDataset.csv', index_col = 0)

attributes, test_attributes, target_label, test_target_label = split_data(data, test_data)

#print(test_target_label)

def knn(attributes, target_label, test_attributes):

	classifier = KNeighborsClassifier(n_neighbors=5)  
	classifier.fit(attributes, target_label) 

	prediction = classifier.predict(test_attributes)

	return prediction


 
def nb(attributes, target_label, test_attributes):

	classifier = GaussianNB()

	classifier.fit(attributes, target_label)
	prediction = classifier.predict(test_attributes)

	return prediction


def tree(attributes, target_label, test_attributes):


	clf_gini = DecisionTreeClassifier(criterion = "gini", random_state = 100,
	                               max_depth=3, min_samples_leaf=5)
	clf_gini.fit(attributes, target_label)

	prediction = clf_gini.predict(test_attributes)

	return prediction


def get_accuracy(test_target_label, prediction):

	count = 0
	for label, predcition_label in zip(test_target_label, prediction):
		#print(label, predcition_label)
		if label == predcition_label:
			count += 1
	return count / len(prediction) * 100



knn_prediction = knn(attributes, target_label, test_attributes)
print("knn")
print(get_accuracy(test_target_label, knn_prediction))

nb_prediction =  nb(attributes, target_label, test_attributes)
print("nb")
print(get_accuracy(test_target_label, nb_prediction))

tree_prediction = tree(attributes, target_label, test_attributes)
print("tree")
print(get_accuracy(test_target_label, tree_prediction))

