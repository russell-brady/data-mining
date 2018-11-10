from time import time 
from sklearn.metrics import f1_score
from data_preparation import *
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.svm import SVC
import numpy as np

data = pd.read_csv('Datasets/relevant_data/cleanedDataset_full.csv', index_col = 0)
test_data = pd.read_csv('Datasets/relevant_data/cleanedTestDataset_full.csv', index_col = 0)

attributes, test_attributes, target_label, test_target_label = split_data_only_hw(data, test_data)

# print(target_label['A'])

# print(target_label)

print(len(attributes), len(test_attributes))

def train_classifier(clf, X_train, y_train):
    ''' Fits a classifier to the training data. '''
    
    # Start the clock, train the classifier, then stop the clock
    start = time()
    clf.fit(X_train, y_train)
    end = time()
    
    # Print the results
    print ("Trained model in {:.4f} seconds".format(end - start))

    
def predict_labels(clf, features, target):
    ''' Makes predictions using a fit classifier based on F1 score. '''
    
    # Start the clock, make predictions, then stop the clock
    start = time()
    y_pred = clf.predict(features)
    
    end = time()
    # Print and return results
    print ("Made predictions in {:.4f} seconds.".format(end - start))
    
    return f1_score(target, y_pred, average=None), sum(target == y_pred) / float(len(y_pred))


def train_predict(clf, X_train, y_train, X_test, y_test):
    ''' Train and predict using a classifer based on F1 score. '''
    
    # Indicate the classifier and the training set size
    print ("Training a {} using a training set size of {}. . .".format(clf.__class__.__name__, len(X_train)))
    
    # Train the classifier
    train_classifier(clf, X_train, y_train)
    
    # Print the results of prediction for both training and testing
    f1, acc = predict_labels(clf, X_train, y_train)
    print (f1, acc)
    print ("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(np.amax(f1) , acc))
    
    f1, acc = predict_labels(clf, X_test, y_test)
    print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(np.amax(f1) , acc))

    # TODO: Initialize the three models (XGBoost is initialized later)

clf_A = LogisticRegression(random_state = 42)
clf_B = SVC(random_state = 912, kernel='rbf')
clf_C = xgb.XGBClassifier(seed = 82)

train_predict(clf_A, attributes, target_label, test_attributes, test_target_label)
print()
train_predict(clf_B, attributes, target_label, test_attributes, test_target_label)
print()
train_predict(clf_C, attributes, target_label, test_attributes, test_target_label)
print()

# TODO: Import 'GridSearchCV' and 'make_scorer'
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer


# TODO: Create the parameters list you wish to tune
parameters = { 'learning_rate' : [0.1],
               'n_estimators' : [40],
               'max_depth': [3],
               'min_child_weight': [3],
               'gamma':[0.4],
               'subsample' : [0.8],
               'colsample_bytree' : [0.8],
               'scale_pos_weight' : [1],
               'reg_alpha':[1e-5]
             }  

# TODO: Initialize the classifier
clf = xgb.XGBClassifier(seed=2)

# TODO: Make an f1 scoring function using 'make_scorer' 
f1_scorer = make_scorer(f1_score,pos_label='H')


# TODO: Perform grid search on the classifier using the f1_scorer as the scoring method
grid_obj = GridSearchCV(clf,
                        scoring=f1_scorer,
                        param_grid=parameters,
                        cv=5)

# TODO: Fit the grid search object to the training data and find the optimal parameters
grid_obj = grid_obj.fit(attributes,target_label)

# Get the estimator
clf = grid_obj.best_estimator_
#print clf

# Report the final F1 score for training and testing after parameter tuning
f1, acc = predict_labels(clf, attributes, target_label)
print(acc)
print("F1 score and accuracy score for training set: {:.4f} , {:.4f}.".format(np.amax(f1) , acc))
    
f1, acc = predict_labels(clf, test_attributes, test_target_label)
print ("F1 score and accuracy score for test set: {:.4f} , {:.4f}.".format(np.amax(f1) , acc))