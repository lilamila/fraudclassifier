
# coding: utf-8

#!/usr/bin/python
import sys
import pickle
import pandas as pd
import numpy as np
import collections, operator
from pprint import pprint
import seaborn as sea
import matplotlib.pyplot as plt

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import dump_classifier_and_data
import explore_data

import warnings
warnings.filterwarnings('ignore')


# full list

"""Task 1: Select features features_list is a list of strings, 
each of which is a feature name. The first feature is "poi"."""

features_list = [
    'poi',
    'salary',
    'to_messages',
    'deferral_payments',
    'total_payments',
    'exercised_stock_options',
    'bonus',
    'restricted_stock',
    'shared_receipt_with_poi',
    'restricted_stock_deferred',
    'total_stock_value',
    'expenses',
    'loan_advances',
    'from_messages',
    'other',
    'from_this_person_to_poi',
    'director_fees',
    'deferred_income',
    'long_term_incentive',
    'from_poi_to_this_person',
    ]

### Load the dictionary containing the dataset

with open('final_project_dataset.pkl', 'r') as data_file:
    data_dict = pickle.load(data_file)

df = pd.DataFrame.from_dict(data_dict, orient='index')
df = df.replace('NaN', np.nan)

### Explore data
print('####### Explore Data #########')

from explore_data import find_highest_paid, count_valid_values
count_valid_values(data_dict)
print ''
print 'Count of people: ', len(data_dict)
poiCount = sum(p['poi'] == 1 for p in data_dict.values())
print 'Count of POIs: ', poiCount


"""Task 2: Remove outliers
"""

### Remove Outliers data_dict dropping: (is this necessary if convert back to_dict at end?)
outliers = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'KAMINSKI WINCENTY J']
for outlier in outliers:
    data_dict.pop(outlier, 0)

### Correct data for Bhatnagar Sanjay

b_sandjay = data_dict['BHATNAGAR SANJAY']
b_sandjay['expenses'] = 137864
b_sandjay['total_payments'] = 137864
b_sandjay['exercised_stock_options'] = 15456734
b_sandjay['restricted_stock'] = 2604490
b_sandjay['restricted_stock_deferred'] = 2604490
b_sandjay['total_stock_value'] = 15456290
b_sandjay['director_fees'] = 'NaN'
b_sandjay['other'] = 'NaN'

### Correct data for B Robert
b_robert = data_dict["BELFER ROBERT"]
b_robert['deferred_income'] = 102500
b_robert['deferral_payments']= 'NaN'
b_robert['expenses'] = 3285
b_robert['directors_fees'] = 102500
b_robert['total_payments']=3285
b_robert['exercised_stock_options']='NaN'
b_robert['restricted_stock_options']=44093

'''Task 3: Create new features'''
for name in data_dict:
    to_messages = data_dict[name]['to_messages']
    from_messages = data_dict[name]['from_messages']
    to_user_from_poi = data_dict[name]['from_poi_to_this_person']
    from_user_to_poi = data_dict[name]['from_this_person_to_poi']
    shared_receipt = data_dict[name]['shared_receipt_with_poi']

    if to_messages != 'NaN' and to_user_from_poi != 'NaN':
        ratio_from_poi = float(to_user_from_poi)/float(to_messages)
    else:
        ratio_from_poi = 'NaN'
    if from_messages != 'NaN' and from_user_to_poi != 'NaN':
        ratio_to_poi = float(from_user_to_poi)/float(from_messages)
    else:
        ratio_to_poi = 'NaN'
    if to_messages != 'NaN' and shared_receipt != 'NaN':
        ratio_shared_receipt = float(shared_receipt)/float(to_messages)
    else:
        ratio_shared_receipt = 'NaN'
 
    data_dict[name]['ratio_from_poi'] = ratio_from_poi
    data_dict[name]['ratio_to_poi'] = ratio_to_poi
    data_dict[name]['ratio_shared_receipt'] = ratio_to_poi


### Feature scaling
minmax_features = {}

for key in data_dict:
    for feature in features_list:
        feature_value = data_dict[key][feature]
        if feature_value != 'NaN':
            if not minmax_features.has_key(feature):
                minmax_features[feature] =                     {"min":feature_value,"max":feature_value}
            elif feature_value > minmax_features[feature]["max"]:
                minmax_features[feature]["max"]=feature_value
            elif feature_value < minmax_features[feature]["min"]:
                minmax_features[feature]["min"]=feature_value

for key in data_dict:
    for feature in features_list:
        feature_value = data_dict[key][feature]
        if feature != "poi" and feature_value != 'NaN':
            feature_value = data_dict[key][feature]
            minmax = minmax_features[feature]
            mrange = minmax['max']-minmax['min']
            feature_value = feature_value - minmax['min']
            feature_value = float(feature_value) / float(mrange)
            data_dict[key][feature] = feature_value

### Store to my_dataset for easy export below.
my_dataset = data_dict
# Convert dictionary to numpy array, converts NaN to 0.0  
# data = featureFormat(my_dataset, features_list, sort_keys = True, remove_all_zeroes = False)
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

# Unsupervised PCA Pipeline - dimensionality reduction, learn patterns of variation in data
from sklearn.pipeline import Pipeline, FeatureUnion #http://scikit-learn.org/stable/modules/pipeline.html
from sklearn.decomposition import PCA #or doPCA
from sklearn.feature_selection import SelectKBest, f_classif

# set number of features
def doKbest(features, labels, k):
    k_select = SelectKBest(f_classif, k=k).fit(features, labels)
    scores = k_select.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features:\n ".format(k)
    pprint (k_best_features)
    print '\n'

def doPCAsplit(features, labels, n_components=''):
    from sklearn.decomposition import PCA
    pca = PCA(n_components).fit(features, labels)
    variances = list(reversed(sorted([round(x,5) for x in pca.explained_variance_ratio_])))
    print "{0} best PCA explained variances:\n ".format(n_components)
    pprint (variances)
    print ''
#     first_pc = pca.components_[0]
#     second_pc = pca.components_[1]
    return pca


doKbest(features, labels, 8)
doPCAsplit(features, labels, n_components=17)


# In[97]:

"""Task 4: Trying a variety of classifiers, using Pipelines for multi-stage
operations. (http://scikit-learn.org/stable/modules/pipeline.html)
*ran this operation first to see how the data cleaning and feature 
creation/extractions process affected the scores.""" 

# Import classifiers
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import AdaBoostClassifier, RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

# Import validation helpers
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score, make_scorer,     precision_score, recall_score, classification_report
from sklearn.cross_validation import train_test_split, StratifiedKFold,     cross_val_score, StratifiedShuffleSplit


def assess_classifier(clf, X, y):
    cv = StratifiedShuffleSplit(y, 35, random_state=42)
    print clf, ' results: '
    print 'accuracy', round(cross_val_score(clf, X, y, cv=cv).mean(), 4)
    print 'precision', round(cross_val_score(clf, X, y,
                             scoring=make_scorer(precision_score),
                             cv=cv).mean(), 4)
    print 'recall', round(cross_val_score(clf, X, y,
                          scoring=make_scorer(recall_score),
                          cv=cv).mean(), 4)
def assess_features(X, y):
    clf = RandomForestClassifier(random_state=42) #  gini, entropy, max_depth, minimum_samples_split
    assess_classifier(clf, X, y)
    print
    clf = AdaBoostClassifier(random_state=42)
    assess_classifier(clf, X, y)
    print
    clf = GaussianNB()
    assess_classifier(clf, X, y)
def assess_forest(X, y):
    clf = RandomForestClassifier(random_state=42)
    assess_classifier(clf, X, y)
def assess_boost(X, y):
    clf = AdaBoostClassifier(random_state=42)
    assess_classifier(clf, X, y)


### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script. Check the tester.py script in the final project
### folder for details on the evaluation method, especially the test_classifier
### function. Because of the small size of the dataset, the script uses
### stratified shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

# Example starting point. Try investigating other evaluation techniques!
# from sklearn.cross_validation import train_test_split
# features_train, features_test, labels_train, labels_test = \
#     train_test_split(features, labels, test_size=0.3, random_state=42)

pca = PCA()
kbest = SelectKBest(f_classif)

forest_clf = RandomForestClassifier(random_state=42)
boost_clf = AdaBoostClassifier(random_state=42)
lr_clf = LogisticRegression(random_state=42)
svc_clf = SVC(class_weight='balanced', random_state=42)
# boost_clf.get_params().keys()

combined_features = FeatureUnion([("pca", PCA()), ("kbest", SelectKBest(f_classif))])


### run ADABOOST

# pipe = Pipeline(steps=[("features", combined_features), 
#         ("boost", AdaBoostClassifier(random_state=42))])
# param_grid_neigh = dict(features__pca__n_components= range(1,21,4),
#                         features__kbest__k= range(1,21,4),
# #                         features__pca__n_components= range(1,len(features_list)),
# #                         features__kbest__k= range(1,len(features_list)),
#                         boost__n_estimators =[1,15, 50, 75],
#                         boost__learning_rate=[0.5,1, 2])  
    
# boost_clf = GridSearchCV(pipe, param_grid=param_grid_neigh, verbose=3,     scoring = 'f1')
# boost_clf.fit(features, labels)

# print "Best Score: ", boost_clf.best_score_
# print "Best Params: ",boost_clf.best_params_
# print ''
# print "Evaluation: "
# assess_classifier(boost_clf.best_estimator_, features, labels)

### run SVC
pipe = Pipeline(steps=[("features", combined_features), 
        ("svc", SVC(class_weight='auto', random_state=42))])
param_grid_neigh = dict(features__pca__n_components= range(1,21,2),
                        features__kbest__k= range(1,21,2),
                        svc__kernel =['rbf'], #'linear'
                        svc__C=[10000, 35000, 50000, 750000, 100000]
                        # svc__C=[100, 1000, 10000, 35000, 50000, 750000, 100000]
                        )  

svc_clf = GridSearchCV(pipe, param_grid=param_grid_neigh, verbose=3, scoring = 'f1')
svc_clf.fit(features, labels)

svc_clf.get_params().keys()

print ''
print "Best Score: ", svc_clf.best_score_
print "Best Params: ",svc_clf.best_params_
print ''
print "Evaluation: "
assess_classifier(svc_clf.best_estimator_, features, labels)


# features_selected_bool = svc_clf.best_estimator_.named_steps['features'].get_support()
# print features_selected_bool

# #### SVC Tuning Results
# Best Score:  0.519802576141
# Best Params:  {'features__pca__n_components': 15, 'svc__kernel': 'rbf', 'svc__C': 10000, 'features__kbest__k': 3}
# Evaluation: 
# Pipeline(steps=[('features', FeatureUnion(n_jobs=1,
#        transformer_list=[('pca', PCA(copy=True, n_components=15, whiten=False)), ('kbest', SelectKBest(k=3, score_func=<function f_classif at 0x1170396e0>))],
#        transformer_weights=None)), ('svc', SVC(C=10000, cache_size=200, class_weight='auto', coef0=0.0,
#   decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
#   max_iter=-1, probability=False, random_state=42, shrinking=True,
#   tol=0.001, verbose=False))])  results: 
# accuracy 0.8552
# precision 0.45
# recall 0.5286

kbest = svc_clf.best_params_.get('features__kbest__k')
doKbest(features, labels, kbest)
n_components = svc_clf.best_params_.get('features__pca__n_components')
doPCAsplit(features, labels, n_components)


from tester import test_classifier

# test_classifier(boost_clf,my_dataset, features_list, folds = 200)
# test_classifier(forest_clf,my_dataset, features_list, folds = 200)
# test_classifier(svc_clf,my_dataset, features_list, folds = 200)

### Task 6: Dump classifier, dataset, and features_list to pkl files
# choose classifier and automatically set params to submit, forest_clf, boost_clf, gnb_clf
clf = svc_clf.best_estimator_ 

# submit final features_list
my_feature_list = features_list

dump_classifier_and_data(clf, my_dataset, my_feature_list)

