# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ###Classifying contracts
# - Choose features
# - Deal with NaNs
# - Tune parameters with cross-validation
# - Test & visualize performance
# 
# ###Classifiers
# - Random forest
# - Binary svm

# <codecell>

import pandas as pd
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC

# <markdowncell>

# ###Load data set and select features for classification
#  - Edit filepath in cell below to point to data set
#  - Edit list of features in cell below to use in classification

# <codecell>

#df_full = pd.read_hdf('../Data/MDB_data/WorldBank/world_bank_combined_clean.hdf5','df')
# df_full = pd.read_csv('../Data/MDB_data/WorldBank/Major_Contract_Awards_clean.csv')

df_full = pd.read_csv('../Data/Procurements/Historic_and_Major_awards_network_features.csv')

#choose columns from data frame to use as features
columns = ['award_amount_usd','competitive']
df_x = df_full[columns]

#choose columns from data frame to use as labels
df_y = df_full['LABELS']

# <markdowncell>

# ###Remove/replace NaNs in features
# - Haven't implemented ... probably data set & feature specific

# <codecell>

#Deal with NaNs as needed
#df.isnull()

# <codecell>

#Create numpy array from dataframe
X = df_x.values
y = df_y.values
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# <codecell>

# generate data for testing pipeline.
# comment out this cell after testing ...
X, y = make_classification(n_samples=5000, n_features=2, n_redundant=0, n_informative=2,
                             n_clusters_per_class=1)
plt.scatter(X[:, 0], X[:, 1], marker='o', c=y)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=0)

# <codecell>

param_grid = [
  {'n_estimators': [5,10,15,20], 'max_features': [1, 2]}
]

scores = ['accuracy']#,'precision', 'recall', ‘roc_auc’]

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print('---')

    rf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring=score)
    rf.fit(X_train, y_train)
    print("Best parameters set found on training set:")
    print(rf.best_estimator_)
    print('---')
    print("Grid scores on development set:")
    for params, mean_score, scores in rf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print('---')

    print("Detailed classification report:")
    y_true, y_pred = y_test, rf.predict(X_test)
    print(classification_report(y_true, y_pred))

# <codecell>

param_grid = [
  {'kernel': ['rbf'], 'gamma': [1e-2,1e-3, 1e-4], 'C': [1, 10, 100, 1000]}
]

scores = ['accuracy']#,'precision', 'recall', ‘roc_auc’]

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print('---')

    svm = GridSearchCV(SVC(probability=True), param_grid, cv=5, scoring=score)
    svm.fit(X_train, y_train)
    print("Best parameters set found on training set:")
    print(svm.best_estimator_)
    print('---')
    print("Grid scores on development set:")
    for params, mean_score, scores in svm.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print('---')

    print("Detailed classification report:")
    y_true, y_pred = y_test, svm.predict(X_test)
    print(classification_report(y_true, y_pred))

# <codecell>

# Compute ROC curve and area the curve
svm_probs = svm.best_estimator_.predict_proba(X_test)
svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_test, svm_probs[:, 1])
svm_roc_auc = auc(svm_fpr, svm_tpr)
rf_probs = rf.best_estimator_.predict_proba(X_test)
rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_test, rf_probs[:, 1])
rf_roc_auc = auc(rf_fpr, rf_tpr)

# Plot ROC curve
plt.clf()
plt.plot(svm_fpr, svm_tpr, label='SVM (auc = %0.2f)' % svm_roc_auc)
plt.plot(rf_fpr, rf_tpr, label='Random Forest (auc = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic example')
plt.legend(loc="lower right")
plt.show()

