#!/usr/bin/python

""" 
    Starter code for exploring the Enron dataset (emails + finances);
    loads up the dataset (pickled dict of dicts).

    The dataset has the form:
    enron_data["LASTNAME FIRSTNAME MIDDLEINITIAL"] = { features_dict }

    {features_dict} is a dictionary of features associated with that person.
    You should explore features_dict as part of the mini-project,
    but here's an example to get you started:

    enron_data["SKILLING JEFFREY K"]["bonus"] = 5600000
    
"""

import pickle
from pprint import pprint
import xml.etree.cElementTree as ET
import re
import operator
from collections import Counter

import pandas as pd
import seaborn as sea
import numpy as np

from feature_format import featureFormat, targetFeatureSplit
from sklearn.feature_selection import SelectKBest, f_classif


def exploratory_answers(enron_data):
    keys = 0
    for key in enron_data:
            keys = keys + 1
    print "Number of People:", keys

    # for key, value in enron_data.iteritems():
    #   print key, value
    #   print "Number of features per person: ", pprint(len(value))

    i = 0
    for key in enron_data:
        # Value of poi field is of type Boolean. So, use True instead of string "True"
        if enron_data[key]['poi'] == True:
            i = i + 1

    print "Number of POIs:", i    
    print "James Prentice's total stock value: $", enron_data["PRENTICE JAMES"]["total_stock_value"]
    print "Wesley Colwell's emails to POIs: ", enron_data["COLWELL WESLEY"]["from_this_person_to_poi"]
    print "Jeffrey Skilling's exercised stock options: ", enron_data["SKILLING JEFFREY K"]["exercised_stock_options"]



def find_highest_paid(enron_data):
    highest_payment = max([enron_data["FASTOW ANDREW S"]["total_payments"], \
                        enron_data["LAY KENNETH L"]["total_payments"], \
                        enron_data["SKILLING JEFFREY K"]["total_payments"]])
    for key, value in enron_data.iteritems():
        if enron_data[key]['total_payments'] == highest_payment:
            print "the highest paid guy was: ", enron_data[key]['total_payments']


def count_valid_values(data_dict):
    """ counts the number of non-NaN values for each feature """
    counts = dict.fromkeys(data_dict.itervalues().next().keys(), 0)
    for record in data_dict:
        person = data_dict[record]
        for field in person:
            if person[field] != 'NaN':
                counts[field] += 1
    print "Valid values: "
    pprint(counts)
    return counts

def get_k_best(data_dict, features_list, k):
    """ runs scikit-learn's SelectKBest feature selection
        returns dict where keys=features, values=scores
    """
    data = featureFormat(data_dict, features_list)
    labels, features = targetFeatureSplit(data)

    k_best = SelectKBest(k=k)
    k_best.fit(features, labels)
    scores = k_best.scores_
    unsorted_pairs = zip(features_list[1:], scores)
    sorted_pairs = list(reversed(sorted(unsorted_pairs, key=lambda x: x[1])))
    k_best_features = dict(sorted_pairs[:k])
    print "{0} best features: {1}\n".format(k, k_best_features.keys())
    return k_best_features


#### Create a new PCA features and modify data_dict
def pca_features(pca_feature_list, my_dataset):

    data = featureFormat(my_dataset, pca_feature_list,remove_NaN=True, remove_all_zeroes=False, remove_any_zeroes=False, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    new_features = pca.fit_transform(features)

    for index, each_person in enumerate(my_dataset.values()):
        each_person['first_pc'] = new_features[index][0]
        each_person['second_pc'] = new_features[index][1]

    return my_dataset

    # first_pc = pca.components_[0]
    # second_pc = pca.components_[1]

    # transformed_data = pca.transform(data)
    # for ii, jj in zip(transformed_data, data):
    #     plt.scatter(first_pc[0]*ii[0], first_pc[1]*ii[0], color="r")
    #     plt.scatter(second_pc[0]*ii[1], second_pc[1], color="c")
    #     #plot original data
    #     plt.scatter(jj[0], jj[1], color="b")

    # plt.xlabel("bonus")
    # plt.ylabel("long-term incentive")
    # plt.show()

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

def doPCAdata(data, n_components=''):
    from sklearn.decomposition import PCA
    pca = PCA(n_components).fit(data)
    variances= list(sorted(pca.explained_variance_ratio_))
    pprint (variances)
    print ''
    return pca

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
    
from sklearn.base import BaseEstimator
from sklearn.base import ClassifierMixin
import numpy as np
import operator

class EnsembleClassifier(BaseEstimator, ClassifierMixin):
    """
    Ensemble classifier for scikit-learn estimators.

    Parameters
    ----------

    clf : `iterable`
      A list of scikit-learn classifier objects.
    weights : `list` (default: `None`)
      If `None`, the majority rule voting will be applied to the predicted class labels.
        If a list of weights (`float` or `int`) is provided, the averaged raw probabilities (via `predict_proba`)
        will be used to determine the most confident class label.

    """
    def __init__(self, clfs, weights=None):
        self.clfs = clfs
        self.weights = weights

    def fit(self, X, y):
        """
        Fit the scikit-learn estimators.

        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]
            Training data
        y : list or numpy array, shape = [n_samples]
            Class labels

        """
        for clf in self.clfs:
            clf.fit(X, y)

    def predict(self, X):
        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        maj : list or numpy array, shape = [n_samples]
            Predicted class labels by majority rule

        """

        self.classes_ = np.asarray([clf.predict(X) for clf in self.clfs])
        if self.weights:
            avg = self.predict_proba(X)

            maj = np.apply_along_axis(lambda x: max(enumerate(x), key=operator.itemgetter(1))[0], axis=1, arr=avg)

        else:
            maj = np.asarray([np.argmax(np.bincount(self.classes_[:,c])) for c in range(self.classes_.shape[1])])

        return maj

    def predict_proba(self, X):

        """
        Parameters
        ----------

        X : numpy array, shape = [n_samples, n_features]

        Returns
        ----------

        avg : list or numpy array, shape = [n_samples, n_probabilities]
            Weighted average probability for each class per sample.

        """
        self.probas_ = [clf.predict_proba(X) for clf in self.clfs]
        avg = np.average(self.probas_, axis=0, weights=self.weights)

        return avg
if __name__ == "__main__":
    enron_data = pickle.load(open("../practice/final_project/final_project_dataset.pkl", "r"))
    exploratory_answers(enron_data)
    find_highest_paid(enron_data)
    count_valid_values(enron_data)