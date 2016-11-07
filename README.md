# Leafer's Enron Fraud Detection

Leafer's Enron Fraud detection, utilizes scikit-learn and machine learning, to build a "person of interest" (POI) identifier that detects and predicts culpable persons. It uses features from financial data, email data, and labeled data. Created by [Marie Leaf](https://twitter.com/mariesleaf).

"You don't want another Enron? Here's your law: If a company, can't explain, in one sentence, what it does... it's illegal."


### Table of contents

* [Summary](#summary)
* [Concepts](#concepts)
* [Creator](#creator)
* [Resources](#resources)
* [Walk Through](#walk through)

### Summary

* [Download the data](https://www.cs.cmu.edu/~./enron/)
* Open the [html version](./final_project/poi_id.html) or [ipynb version](./final_project/poi_id.ipynb)of my workthrough.

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. This project builds a person of interest (POI) identifier based on financial and email data made public as a result of the Enron scandal. 

POIs who were indicted, reached a settlement or plea deal with the government, or testified in exchange for prosecution immunity.

### Concepts

Support Vector Machines, Ensemble Learners.  
Decided to just run nbconvert instead of [automatically](http://protips.maxmasnick.com/ipython-notebooks-automatically-export-py-and-html) creating .py file from .ipynb  

### Creator

**Marie Leaf**

* [Follow @mariesleaf](https://twitter.com/mariesleaf)
* [Github](https://github.com/mleafer)

### Resources

* [Ensemble Classifiers](http://scikit-learn.org/stable/modules/ensemble.html)
* [AdaBoost](http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.AdaBoostClassifier.html)
* [AdaBoost Weights](http://stackoverflow.com/questions/18054125/how-to-use-weight-when-training-a-weak-learner-for-adaboost)
* [Tuning Classifiers](https://discussions.udacity.com/t/tuning-the-chosen-classifier/161560)
* [Using GridsearchCV with Adaboost and Random Forest](http://stackoverflow.com/questions/32210569/using-gridsearchcv-with-adaboost-and-decisiontreeclassifier)
* [Pipeline and ensembles](http://sebastianraschka.com/Articles/2014_ensemble_classifier.html)
* [Pipeline chaining estimators](http://scikit-learn.org/stable/modules/pipeline.html)
* [Plot PCA chaining in Pipeline (Autos)](http://scikit-learn.org/stable/auto_examples/plot_digits_pipe.html#example-plot-digits-pipe-py)
* [Plot PCAs (Autos)](http://scikit-learn.org/stable/auto_examples/decomposition/plot_pca_iris.html)
* [More PCA](https://plot.ly/ipython-notebooks/principal-component-analysis/)
* [Feature Scores from Pipeline](https://discussions.udacity.com/t/how-to-find-out-the-features-selected-by-selectkbest/45118/3)
* [Feature Stacker](http://scikit-learn.org/stable/auto_examples/feature_stacker.html)
* [ScoreTracker](https://docs.google.com/spreadsheets/d/17Dfc6YY4BEzsf_n7j4Rp8hUX8GlvsQmZPr80iMbHKEA/edit#gid=0) (own personal tool)

### Walk Through

---
*Goals and why machine learning? Dataset background and rationale for choosing it. Discussion of outlier identification and handling*

This project trains a machine learning classifier over the varied corpus of discrete, and at times seemingly disparate Enron data, to accurately predict persons-of-interests in detecting fraud. While the Enron case happened in the past, this classifier could be potentially applied to companies with similar datasets to detect fraudulent behavior of employees. Machine learning techniques are employed to vigorously train and rigourously test various algorithms to process a large amount of data - more data than one person can accurately process. Techniques to perform dimensionality reduction of the data, ie see patterns of data variance to draw out new predictive features is also used through Principle Component Analysis.

The aggregated Enron email and financial dataset is stored in a dictionary, where each key in the dictionary is a person’s name and the value is a dictionary containing all the features of that person. The email and finance (E+F) data dictionary is stored as a pickle file, which is a handy way to store and load python objects directly. The features in the data fall into three major types, namely financial features, email features and POI labels.

Valid values: 
{'bonus': 82,
 'deferral_payments': 39,
 'deferred_income': 49,
 'director_fees': 17,
 'email_address': 111,
 'exercised_stock_options': 102,
 'expenses': 95,
 'from_messages': 86,
 'from_poi_to_this_person': 86,
 'from_this_person_to_poi': 86,
 'loan_advances': 4,
 'long_term_incentive': 66,
 'other': 93,
 'poi': 146,
 'restricted_stock': 110,
 'restricted_stock_deferred': 18,
 'salary': 95,
 'shared_receipt_with_poi': 86,
 'to_messages': 86,
 'total_payments': 125,
 'total_stock_value': 126}

Count of people:  146
Count of POIs:  18

Outliers removed: 
('KAMINSKI WINCENTY J') - an employee that was a mass emailer, removed to ensure the validity of email features weren't skewed.
('THE TRAVEL AGENCY IN THE PARK') - not an employee! (I don't think)
('TOTAL') - skewed all the data
(['email_address']) - not an outlier, but an obvious addition of noise to the model, as it acts as a redundant identification variable for each case. 

The strongest outlier in the dataset was the row containing the total of all the columns. The algorithm predictions improved significantly after this row was removed. There was also an employee who sent more emails than double the next highest number. This entity was removed to prevent it from skewing the average amount of emails sent out as a possible marker.

In terms of setting up the data to provide the most predictive power, I dropped the email address column, as this is a redundant identification variable.

One of the biggest issues, and hindrances for me finishing this project successfully, was imputing the median and mean values of the variables for NaN data. There was so much missing data, that this probably skewed the efficacy of the data's predictive power. Once I replaced the NaN values with '0', I started to hit considerably better results aross the board for all classifiers. I really learned how important the ratio of missing values is, in determining how to proceed with imputing vs. substituting data.

---

*Final features used in POI classifier, and their selection process. Discussion of scaling and feature engineering + testing. Discussion of algorithm used, and feature importances and scores and reasons for choices.*

I decided to code up a new feature that considers the ratio of messages *sent to* and ratio of messages that *came from* a POI. As corruption is rarely a one person endeavor, this feature would indicate a network exhibit through email exchange. From the EDA scatterplot, we can see that if a person has a ratio of messages TO a POI of less than ~0.18 and a ratio FROM a POI of less than ~0.25, then  they probably are not a POI. This new feature added to the predictive value of our model.

From my time working at a investment advisory firm, looking at public company incentive structures, I used my domain knowledge and understanding of human psychology to explore engineering new long vs. short term incentive structure features. 

After much parsing through the different iterations of engineering long and short term incentive features, there did not seem to be a simple configuration that could be added to the model. The ratios of (long term payoff(restricted stock and long term incentives):expenses plus bonuses) plotted against ratios of (long term payoffs:exercised stock options) could possibly be used as decision tree predictors, so those features were kept. After testing the models with these LT/ST features, however, they actually brought down the accuracy, precision, and recall across all three classifiers. So I will excluded these from my final project. I am open to the possibility that this behavior may change with larger datasets.

I recorded the scores here:  
[Score Tracker](
https://docs.google.com/spreadsheets/d/17Dfc6YY4BEzsf_n7j4Rp8hUX8GlvsQmZPr80iMbHKEA/edit?usp=sharing)

|1- Initial |2 - drop outliers  |3 - new email ratio features   |3 - new LT/ST feature|
|-----|----|-------|----|
|Random Forest|  |  | |
|0.8472 |0.8541 |0.8629 |0.861|
|0.1452 |0.1991 |0.2239 |0.1916|
|0.1015 |0.126  |0.143  |0.115|
|Adaboost|  |  | |          
|0.8449 |0.8415 |0.8511 |0.8207|
|0.3407 |0.3189 |0.3687 |0.259|
|0.3045 |0.2905 |0.335  |0.265|
|Gaussian NB|  |  | |           
|0.3819 |0.7391 |0.739  |0.5746|
|0.1702 |0.2698 |0.2696 |0.2035|
|0.831  |0.3805 |0.38   |0.6|

Before I learned how to use SelectKBest in Pipeline and GridsearchCV, I spent a lot of time dropping unimportant features upon each parameter tune of my algorithms (evidenced by my run-off code). This very effectively taught me the value of using Pipeline to automate and clean up this process. I got dramatically different results using marginally different amounts of features. I'm not sure how much this is a function of the limited dataset or the process of feature selecting in general. Looking forward to working on larger datasets.

In sum, I used univariate (not recursive) feature selection with the SelectKBest algorithm.

3 best features used in the submitted SVC classifier:

| Feature                 | Importance |
| :-----------------------| ---------  |
|'total_stock_value'      | 22.299|
|'exercised_stock_options'| 22.146|
|'bonus'                  | 20.299|
|'salary'| 18.357|
|'long_term_incentive'| 9.729|

17 best PCA explained variances:
 
[0.33458,
 0.22521,
 0.1098,
 0.07736,
 0.06084,
 0.03863,
 0.03061,
 0.02366,
 0.02231,
 0.01744,
 0.0168,
 0.01502,
 0.01031,
 0.00841,
 0.0047,
 0.00242,
 0.00185]

I tried using the [minmaxscaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html), but this did not change the scores of the Forest and Adaboost classifiers. However, I added feature scaling with a custom function outside of the pipeline when I implemented the SVC algorithm, as it requires feature scaling. This ensures that the variables are sclaed by the min/max of the whole set, not just the training/testing set.

When using PCA or SVM, it can also be helpful to center the data using something like [StandardScaler](http://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.StandardScaler.html)

---
*Final algorithm choice and rationale. Discussion of other algorithms tested and comparative model performance.*


*Random Forest (bagging voting):* 
The Random Forest ensemble learner is a series of decision trees built on randomized subsets of the data. 
At each node, a randomized subset of variables is evaluated to find the best split. This creates a “forest” of independent, randomized trees. The predictions of all the trees are averaged to assign a predicted label. 

**Adaboost:**
AdaBoost is by default an ensemble of decision trees (although the base classifier can be altered), but the trees are not independent. First, a single tree is fit using all of the data. Incorrectly assigned points are then given a higher weight (boosted) and the process is repeated. By default, adaboost uses single feature trees of depth 1 (decision stumps)

**Adaboost Tuning Results:**
Best Score:  0.407490396927
Best Params:  {'features__pca__n_components': 17, 'boost__learning_rate': 0.5, 'boost__n_estimators': 50, 'features__kbest__k': 17}

Evaluation: 
Pipeline(steps=[('features', FeatureUnion(n_jobs=1,
       transformer_list=[('pca', PCA(copy=True, n_components=17, whiten=False)), ('kbest', SelectKBest(k=17, score_func=<function f_classif at 0x11702b050>))],
       transformer_weights=None)), ('boost', AdaBoostClassifier(algorithm='SAMME.R', base_estimator=None,
          learning_rate=0.5, n_estimators=50, random_state=42))])  results: 
accuracy 0.859
precision 0.4524
recall 0.3143

**Support Vector Classifier:**
Best Score:  0.538581918864
Best Params:  {'features__pca__n_components': 17, 'svc__kernel': 'rbf', 'svc__C': 35000, 'features__kbest__k': 5}

Evaluation: 
Pipeline(steps=[('features', FeatureUnion(n_jobs=1,
       transformer_list=[('pca', PCA(copy=True, n_components=17, whiten=False)), ('kbest', SelectKBest(k=5, score_func=<function f_classif at 0x1164e38c0>))],
       transformer_weights=None)), ('svc', SVC(C=35000, cache_size=200, class_weight='auto', coef0=0.0,
  decision_function_shape=None, degree=3, gamma='auto', kernel='rbf',
  max_iter=-1, probability=False, random_state=42, shrinking=True,
  tol=0.001, verbose=False))])  results: 
accuracy 0.8686
precision 0.4857
recall 0.5

The final classifier submitted was a SVC with the parameters outlined above. 

For this project, I wanted to learn a huge range of classifiers to build up a repertoire for future projects. I preliminarily tried KNN, Support Vector Machine, and Gaussian Naive Bayes, but decided to focus on tuning and testing Adaboost and Random Forest ensemble classifiers. These classifiers have proven to be very powerful at Kaggle competitions, and I love exploring everything related to information entropy. Even though they are not primarily known for classification tasks, and wouldn't work well in integrating new data, they work well for the purposes of this project. I would love to explore which algorithms are best for introducing new data to the classifier, or in other words are best for continuous integration.

---
*Notes on parameter tuning, and risks of poor tuning* 

Tuning parameters of an algorithm is much like tuning a music string between flat and sharpe; one tunes the algorithm between bias and variance to get a better score for a metric that one is optimizing for (ie. accuracy, precision or recall). It may be the no. of nodes or the minimum amount of samples needed for a decision tree split; it may be tuning the *learning rates* of weak learners' in AdaBoost; Or, it may be toggling the number of clusters in a Kmeans.

To systematically and automatically iterate through all the scores of varying parameter combinations, the GridSearchCV, paired with a pipeline to include the principal components in the model, is an extremely helpful tool.

---
*Notes on validation and my validation strategy*

Validation is performed to ensure that the classifier has strong external validity. A classic mistake it aims to avoid is overfitting the data, where the classifier is tuned to closely to only one case of the training/testing split of the data. 

To avoid this, I used the [StratifiedShuffleSplit](http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html), which is a merged cross-validation object of StratifiedKFold and ShuffleSplit, which returns stratified randomized folds.   

---
*Evaluation metrics chosen and their performance in a human-readable interpretation.*

**Accuracy:**
As the dataset in this project is very small and the ratio of negatives to positives is highly skewed (18 POIs out of 146 total cases), a classifier that predicted only non-POIs as output, would obtain an accuracy score of 87.4%. In this regard, accuracy is not a strong validation metric for our classifier. 

Instead, precision and recall were used to evaluate the information gain and reliability of the classifier to identify POIs.

**Precision(TruePos/TruePos+FalsePos):**

The precision value scores the ratio of true positive estimations over all positive estimations (true or false). In this project, it's the probability that a POI identified by the estimator (a positive) actually is a POI (a true positive). The probability of randomly sampling a POI from the dataset is about 0.12 and the probability of the algorithm to predict a true positive (out of all the positives it suggests) is on average 0.39. This suggests we are gaining information when using the estimator, and it is a *sufficient* hueristic for this goals of this project, but I believe a more offensive metric is a *necessary* hueristic to address fraud detection.

**Recall (TruePos/TruePos+FalseNeg):**
The [adaboosted decision tree?] achieved a recall value of [0.308], suggesting that 69.2% of POIs go undetected. One could argue that this number is too low, as there is a lower cost to proving a non-corrupt person innocent compared to the cost of missing a true POI who would get away with fraudulently obtaining millions of dollars, however 

In this context, recall is more important than precision (and accuracy) so re-tuning the algorithm to yield a better recall value at the cost of precision would be a more effective system in production to help in fraud detection. 

Given a larger dataset, perhaps fraud corpuses across companies, accuracy may prove to be a more relevant metric, however the negative to positive ratios would (hopefully) stay highly skewed.