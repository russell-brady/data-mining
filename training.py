from sklearn.neighbors import KNeighborsClassifier
from data_preparation import *
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

data = pd.read_csv('Datasets/relevant_data/cleanedDataset_full.csv', index_col = 0)
test_data = pd.read_csv('Datasets/relevant_data/cleanedTestDataset_full.csv', index_col = 0)

#attributes1, test_attributes1, target_label1, test_target_label1 = split_data(data, test_data)

#new_attributes, new_test_attributes, new_target_label, new_test_target_label = split_data_only_hw(data, test_data)

bayesData = pd.read_csv('Datasets/relevant_data/forBayes2.csv', index_col = 0)

x_all = bayesData.drop(['FTR'], 1)
y_all = bayesData['FTR']
X_train, X_test, y_train, y_test = train_test_split(x_all, y_all, 
                                                    test_size = 380,
                                                    stratify = y_all)



#print(new_target_label)

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



# knn_prediction1 = knn(attributes1, target_label1, test_attributes1)
# print("knn -- home Away Draw")
# print(get_accuracy(test_target_label1, knn_prediction1))

# nb_prediction1 =  nb(attributes1, target_label1, test_attributes1)
# print("nb home Away Draw")
# print(get_accuracy(test_target_label1, nb_prediction1))

# tree_prediction1 = tree(attributes1, target_label1, test_attributes1)
# print("tree home Away Draw")
# print(get_accuracy(test_target_label1, tree_prediction1))


# knn_prediction = knn(new_attributes, new_target_label, new_test_attributes)
# print("knn")
# print(get_accuracy(new_test_target_label, knn_prediction))

# nb_prediction =  nb(new_attributes, new_target_label, new_test_attributes)
# print("nb")
# print(get_accuracy(new_test_target_label, nb_prediction))

# tree_prediction = tree(new_attributes, new_target_label, new_test_attributes)
# print("tree")
# print(get_accuracy(new_test_target_label, tree_prediction))

knn_prediction1 = knn(X_train, y_train, X_test)
print("knn -- home Away Draw")
print(get_accuracy(y_test, knn_prediction1))

nb_prediction1 =  nb(X_train, y_train, X_test)
print("nb home Away Draw")
print(nb_prediction1)
print(get_accuracy(y_test, nb_prediction1))

tree_prediction1 = tree(X_train, y_train, X_test)
print("tree home Away Draw")
print(get_accuracy(y_test, tree_prediction1))
