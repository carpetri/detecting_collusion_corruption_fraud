# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <markdowncell>

# ###Classification Pipeline
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
import numpy as np
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, roc_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, scale
import matplotlib.pyplot as plt

# <markdowncell>

# ###Load data set and select features for classification

# <codecell>

#df_full = pd.read_csv('../Data/Investigations/IWF_company_matched.csv')
df_full = pd.read_csv('../Data/Investigations/IWF_country_matched.csv')

# <codecell>

# sum(df_full.isnull()) / df_full.shape[0]


# <codecell>

#choose columns from data frame to use as features
columns = [
    'num_contracts',
    'award_amount_usd',
    'total_contracts_country',
#     'percent_contracts_country',
    'total_award_amount_country',
#     'percent_award_amount_country', 
    'percent_competitive', 
    'business_disclosure_index',
    #'firms_competing_against_informal_firms_perc',
    'payments_to_public_officials_perc',
    'legal_rights_index',
    'time_to_enforce_contract',
    'bribes_to_tax_officials_perc',
    'gdp_per_capita',
    'primary_school_graduation_perc',
    'gini_index',
    'unemployment_perc'
]
df_x = df_full[columns]

#choose columns from data frame to use as labels
df_y = df_full['substantiated']

# <markdowncell>

# ###Remove/replace NaNs in features
# - Data set & feature specific

# <codecell>

#For now, replace NaNs with mean from rest of columns
df_x = df_x.fillna(df_x.mean())

# <codecell>

X_train, X_test, y_train, y_test = train_test_split(
    df_x, df_y, test_size=0.25, random_state=0)

# <markdowncell>

# ##Train Random Forest classifier.
# - Tune parameters through cross-validation

# <codecell>

param_grid = [
  {'n_estimators': [5,10,15,20]}
]

scores = ['accuracy','precision', 'recall', 'roc_auc']

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

clf = rf.best_estimator_
#clf.feature_importances_

# <markdowncell>

# ### Feature Scaling and Normalization

# <codecell>

df_x_scaled = scale(df_x)
X_train, X_test, y_train, y_test = train_test_split(
    df_x_scaled, df_y, test_size=0.25)#, random_state=0)

# <codecell>

param_grid = [
    {'kernel': ['rbf'], 'gamma': [1,1e-1,1e-2,1e-3], 'C': [1, 10, 100, 250]},
    {'kernel': ['linear'], 'C': [1, 10]}
]

scores = ['accuracy','precision', 'recall', 'roc_auc']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print('---')

    svm = GridSearchCV(SVC(probability=True), param_grid, cv=10, scoring=score, verbose=True)
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

# Compute ROC curve and area the curve
# svm_probs = svm.best_estimator_.predict_proba(X_train)
# svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_train, svm_probs[:, 1])
# svm_roc_auc = auc(svm_fpr, svm_tpr)
# rf_probs = rf.best_estimator_.predict_proba(X_train)
# rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_train, rf_probs[:, 1])
# rf_roc_auc = auc(rf_fpr, rf_tpr)

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

