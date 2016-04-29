# -*- coding: utf-8 -*-
# Evaluation of classifiers
# ==
# 
# Inputs
# --
# * classifier (as a pickle?
# * a way to get from the classifier a score/ranking of each sample
# 
# Description of approach
# --
# 0. Split the overlap between investigations and procurements into 2/3 train, 1/3 test.
# 
# **Non-graphical method**
# 1. Grab a sample of X random procurements that are not in investigations.
#     - Use classifier to classify the 1/3 test set, and these random procurements. It is our assumption that the number of builty parties in the random procurements set is <<< the number in investigations, so the classifier should predict a lower number guilty in the random procurements than 1/3 train set.
# 
# **Graphical methods**
# 
# 2. Combine the 1/3 test set with all procurements and classify everything. 
#     - If you rank all samples on their score/risk (plotted on x-axis) and on the y-axis the percent of the 1/3 test set that has a risk score below x, the line should not be linear.
#     - On the x-axis plot percent of 1/3 test set that has been captured after examining y% of all samples (y-axis). Looking through a small part of all samples (ranked according to risk) should quickly capture high number of 1/3 test set samples.
# 
# 
# Concrete steps
# --
# 1. Load classifier (from a pickle?)
# 2. ???
# 3. Profit!!!
# 

import pandas as pd
import seaborn # this changes matplotlib defaults to make the graphs look cooler!
import pickle 
from sklearn.cross_validation import train_test_split
from sklearn import svm
from sklearn.cross_validation import KFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.grid_search import GridSearchCV
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc
from sklearn.cross_validation import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, scale
import matplotlib.pyplot as plt
import numpy as np

# load contracts data
contracts = pd.read_csv('../Data/Procurements/awards_with_features.csv', index_col=0, low_memory=False)

# load network features
network_country = pd.read_csv('../Data/Procurements/Historic_and_Major_awards_network_features_Country2.csv', index_col=0, low_memory=False)
network_global = pd.read_csv('../Data/Procurements/Historic_and_Major_awards_network_features.csv', index_col=0, low_memory=False)

# kick out contracts with no supplier to match
contracts = contracts[contracts['canonical_name'].notnull()]

# load investigations
investigations = pd.read_csv('../Data/Investigations/investigations.csv', index_col=0, low_memory=False)
investigations['guilty_or_unknown'] = \
  np.logical_and(  investigations.outcome_of_overall_investigation_when_closed.notnull(), \
                investigations.outcome_of_overall_investigation_when_closed != 'Unfounded') 

# this should assign True = Investigated, False = Otherwise

# group by canonical_name and country to remove duplicates
def reduce_to_one(my_list):
    return my_list.unique()

aggregations = {
    'guilty_or_unknown':'sum',
    'unique_id': reduce_to_one
}
investigations = investigations.groupby(by=['canonical_name', 'country'], as_index=False).agg(aggregations)

# drop investigations that where outcome of overall investigation is Unfounded or missing
investigations = investigations[investigations['guilty_or_unknown'] > 0]


# clean up network features data set
net_global_features = network_global.columns[78:].tolist()
net_global_features.extend(['unique_id'])
network_global = network_global[net_global_features]

net_country_features = network_country.columns[4:].tolist()
network_country = network_country[net_country_features]


# for col in contracts.columns: print col

# create full data set
df = pd.merge(left=contracts,
                   right=investigations,  
                   left_on=['canonical_name', 'buyer_country'], #, 'fiscal_year'],
                   right_on=['canonical_name', 'country'], #, 'fy_complaint_opened'],
                   how='left') # this makes sure that we keep all procuremenets, whether they have a matching investigation or not

df.rename(columns={'unique_id_x':'unique_id_contracts', 'unique_id_y':'unique_id_invests'}, inplace=True)

df = df.merge(right=network_global,
              left_on='unique_id_contracts',
              # left_on='unique_id',
              right_on='unique_id')

df = df.merge(right=network_country,
              left_on='unique_id_contracts',
              right_on='unique_id')
del df['unique_id_x']
del df['unique_id_y']
del df['country']
df['overlap'] = df['unique_id_invests'].notnull()


# Select features
features = df.columns.tolist()
remove_list = [
    'buyer',
    #'buyer_country',
    #'project_id',
    #'unique_id_contracts',
    'unique_id_invests',
    #'major_sector_clean',
    #'canonical_name',
    'guilty_or_unknown',
    'Supplier_Average_Distance_Investigated_Suppliers_Contemporary_Global',
    'Project_Average_Distance_Investigated_Suppliers_Contemporary_Global',
    'Supplier_Average_Distance_Investigated_Suppliers_Cumulative_Global',
    'Project_Average_Distance_Investigated_Suppliers_Cumulative_Global',
    'Supplier_Average_Distance_Investigated_Projects_Contemporary_Global',
    'Project_Average_Distance_Investigated_Projects_Contemporary_Global',
    'Supplier_Average_Distance_Investigated_Projects_Cumulative_Global',
    'Project_Average_Distance_Investigated_Projects_Cumulative_Global',
    'Supplier_Average_Distance_Investigated_Suppliers_Contemporary_Country',
    'Project_Average_Distance_Investigated_Suppliers_Contemporary_Country',
    'Supplier_Average_Distance_Investigated_Suppliers_Cumulative_Country',
    'Project_Average_Distance_Investigated_Suppliers_Cumulative_Country',
    'Supplier_Average_Distance_Investigated_Projects_Contemporary_Country',
    'Project_Average_Distance_Investigated_Projects_Contemporary_Country',
    'Supplier_Average_Distance_Investigated_Projects_Cumulative_Country',
    'Project_Average_Distance_Investigated_Projects_Cumulative_Country',
    'Supplier_Degree_Centrality_Contemporary_Country',
    'Supplier_Degree_Centrality_Cumulative_Country',
    'Supplier_Degree_Centrality_Contemporary_Global',
    'Supplier_Degree_Centrality_Cumulative_Global',
]

for feature in remove_list: features.remove(feature)
remove_list = []

#remove project-supplier and supplier-project features
for col in features:
    if 'Project' in col and 'Supplier' in col:
       remove_list.append(col)
for feature in remove_list: features.remove(feature)

#remove all "distance to investigated" features
remove_list = []
for col in features:
    if 'Investigated' in col:
       remove_list.append(col)
for feature in remove_list: features.remove(feature)

df = df[features]


# Select features
features = df.columns.tolist()
remove_list = [
    'buyer_country',
    'project_id',
    'unique_id_contracts',
    'major_sector_clean',
    'canonical_name'
]

for feature in remove_list: features.remove(feature)


# # Missing data report
# print 'PERCENT MISSING\n--------------------------------------------'
# percent_missing = []
# for feature in features:
#     percent_missing.append([feature, float(sum(df[feature].isnull())) / df.shape[0] * 100])
#     print feature, float(sum(df[feature].isnull())) / df.shape[0] * 100

# print '\n\n'

# print 'PERCENT INFINITY\n--------------------------------------------'
# for feature in features:
#     print feature, float(sum(np.isinf(df[feature].astype(float)))) / df.shape[0] * 100


# Impute missing values
# Impute separately for overlap and non-overlap

# Columns with zero replacing nans
zero_nan_cols = [
    'Supplier_Neighbor_Intensity_Contemporary_Global',
    'Project_Neighbor_Intensity_Contemporary_Global',
    'Supplier_Neighbor_Intensity_Cumulative_Global',
    'Project_Neighbor_Intensity_Cumulative_Global',
    'Supplier_Neighbor_Intensity_Contemporary_Country',
    'Project_Neighbor_Intensity_Contemporary_Country',
    'Supplier_Neighbor_Intensity_Cumulative_Country',
    'Project_Neighbor_Intensity_Cumulative_Country'
]
df[zero_nan_cols] = df[zero_nan_cols].fillna(value=0)

# Columns with columnwise means replacing nans
mean_nan_cols = df[features].columns.tolist()
for col in zero_nan_cols:
    mean_nan_cols.remove(col)
    
for col in mean_nan_cols:
    mean = df[col].mean()
    df[col] = df[col].fillna(value=mean)

for col in mean_nan_cols:
    overlap_mean = df.ix[df['overlap'],col].mean()
    nonoverlap_mean = df.ix[~df['overlap'],col].mean()
    df.ix[df['overlap'],col] = df.ix[df['overlap'],col].fillna(value=overlap_mean)
    df.ix[~df['overlap'],col] = df.ix[~df['overlap'],col].fillna(value=nonoverlap_mean)


# Convert all columns to float
for col in df[features].columns:
    df[col] = df[col].astype(float)


# Replace Inf with (highest columnwise non-Inf)*1.1
for col in df[features].columns:
    df[col].replace(np.inf, -1, inplace=False)
    max_value = max(df[col])*1.1
    df[col].replace(-1, max_value, inplace=False)

# df= df.dropna()


# Take random sample of non-overlapping contracts to balance classes
df.dropna()
print df.shape
from random import sample
random_inds = sample(df[~(df['overlap'].astype(bool))].index.values, 4000)
df_nonoverlap_random = df[features].ix[random_inds]
df_overlap = df[features][df['overlap'].astype(bool)]
full_data = df_overlap.append(df_nonoverlap_random, ignore_index=True)
# full_data=df

# Sort by year
full_data = full_data.sort('fiscal_year')
del full_data['fiscal_year']
labels = full_data['overlap']
del full_data['overlap']


# df_overlap.shape


X_train, X_test = full_data[:-2000].as_matrix(), full_data[-2000:].as_matrix()
y_train, y_test = labels[:-2000].as_matrix(), labels[-2000:].as_matrix()

# Clean-up memory, because the files we loaded earlier are huge and take up a lot of ram
# # this frees up about half a gig
del investigations 
del contracts
del network_country
del network_global


# Composition of train and test sets

# title('TRAIN set labels', fontsize=18)
# vcs = pd.Series(y_train).value_counts()
# plt.pie(vcs, labels=['0 (n = %d)' % (vcs.values[0]), '1 (n = %d)' % (vcs.values[1])]);
# vcs


# title('TEST set labels', fontsize=18)
# vcs = pd.Series(y_test).value_counts()
# plt.pie(vcs, labels=['0 (n = %d)' % (vcs.values[0]), '1 (n = %d)' % (vcs.values[1])]);
# vcs


# #Decision Tree Classifier (2-class)


from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier()
tree_clf = tree_clf.fit(X_train, y_train)
print classification_report(y_test, tree_clf.predict(X_test))


# Examine leaves of the tree
# --


# for ind in argsort(tree_clf.feature_importances_)[::-1]:
#     print '%0.2f' % tree_clf.feature_importances_[ind], features[ind]

# from sklearn.externals.six import StringIO
# with open("decision_tree_classifier.dot", 'w') as f:
#     f = tree_clf.export_graphviz(tree_clf, out_file=f)


# #Random Forest


from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

param_grid = [{
#     'max_features':[None, 'auto'],
#     'max_depth':[None, sqrt(len(features))],
    'n_estimators': [100] # of these, the more the better

}] 

scores = ['roc_auc']

for score in scores:
    print("# Tuning hyper-parameters for %s" % score)
    print('---------------')

    rf_clf = GridSearchCV(RandomForestClassifier(), param_grid, cv=5, scoring=score) # cross-validation takes a while!!!
    rf_clf.fit(X_train, y_train)
    print("Best parameters set found on training set:")
    print(rf_clf.best_estimator_)
    print('---')
    
    print("Grid scores on development set:")
    for params, mean_score, scores in rf_clf.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print('---')

    print("Detailed classification report:")
    print(classification_report(y_test, rf_clf.predict(X_test)))


y_pred = rf_clf.predict(X_test)


pd.Series(y_pred).value_counts()


for ind in argsort(rf_clf.best_estimator_.feature_importances_)[::-1]:
    print '%0.2f' % rf_clf.best_estimator_.feature_importances_[ind], features[ind]


feature_importances = []
for ind in argsort(rf_clf.best_estimator_.feature_importances_)[::-1]:
    feature_importances.append([features[ind], rf_clf.best_estimator_.feature_importances_[ind]])
feature_importances = pd.DataFrame(feature_importances)
feature_importances.columns = ['feature','importance']

percent_missing = pd.DataFrame(percent_missing)
percent_missing.columns = ['feature', 'percent_missing']

feature_df = pd.merge(feature_importances, percent_missing, on='feature', how='left')


scatter(feature_df['importance'], feature_df['percent_missing'])


feature_importances.shape


# Generate ROC curve
# Compute ROC curve and area under the curve
rf_probs = rf_clf.best_estimator_.predict_proba(X_test)
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
plt.plot(rf_fpr, rf_tpr, label='Random Forest (auc = %0.2f)' % rf_roc_auc)
plt.plot([0, 1], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.xlabel('% Uninvestigated Contracts Marked as "Should be investigated"')
plt.ylabel('% Investigated Contracts Caught by Model')
plt.title('True Positive Rate vs. False Positive Rate')
plt.legend(loc="lower right")
plt.show()


# Generate precision-recall curve
rf_probs = rf_clf.best_estimator_.predict_proba(X_test)
rf_precision, rf_recall, rf_thresholds = precision_recall_curve(y_test, rf_probs[:, 1])
#rf_PR_auc = auc(rf_precision, rf_recall, reorder=True)

# Compute ROC curve and area the curve
# svm_probs = svm.best_estimator_.predict_proba(X_train)
# svm_fpr, svm_tpr, svm_thresholds = roc_curve(y_train, svm_probs[:, 1])
# svm_roc_auc = auc(svm_fpr, svm_tpr)
# rf_probs = rf.best_estimator_.predict_proba(X_train)
# rf_fpr, rf_tpr, rf_thresholds = roc_curve(y_train, rf_probs[:, 1])
# rf_roc_auc = auc(rf_fpr, rf_tpr)

# Plot precision_recall curve
plt.clf()
plt.plot( rf_recall, rf_precision, label='Random Forest')
plt.plot([1, 0], [0, 1], 'k--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.1])
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.legend(loc="lower left")
plt.show()


# #Use the random-forest classifier rc_clf to give EVERY procurement a risk score
# 
# Note: this is mainly for visualization purposes, because we made the classifier to use past data to predict future data
# --


for col in full_data.columns: print col


x = df[~df['overlap'].astype(bool)]
y = rf_clf.predict_proba(x[full_data.columns])[:,1]
# scatter(arange(len(y)), y)


df[['unique_id_contracts','risk']].to_csv('contract_risk.csv')


x['risk'] = y


x_sorted = x.sort(columns='risk', ascending=False)


x_sorted.columns


def printer(row):
    print row['canonical_name'], row['fiscal_year'], row['buyer_country'], row['major_sector_clean']

x_recent = x_sorted[x_sorted['fiscal_year']>=2010]
#x_recent.head(n=20).apply(printer, axis=1)
x_recent.head(n=20)[['canonical_name','fiscal_year','buyer_country','major_sector_clean']]


# use the imputer
probs = rf_clf.predict_proba(df[features].dropna())[:,1]
min(probs), max(probs)


df.shape, contracts.shape


# #NOT USING ANY OF THE CODE BELOW, FOR NOW...


# #One-class SVM
# 
# Notes
# --
# - Can't optimize parameters?:
# 
#     - The problem of doing this with a one-class SVM is *if the recall is our only score-function* (and this is the case for a 
# one-class situation), the parameters will be increased/compacted in such a way that the decision boundary expands/contracts 
# until recall is 1.00, at which point our classifier is doing no work (?)
# Basically if there's nothing constraining the decision boundary, it will expand until it captures all points, which will perform
# terribly on the random procurements <--- assumption!
# 
# - using only 2 country-level features, on training set recall is ~60%, on test set it's 50% (i.e. totally random)
#     - using network features in addition makes recall even worse


# Tuning hyper-parameters with OneClassSVM with grid search
# --


unique(X_train == np.inf)


# Note: one-class svm calls everything outside its boundary '-1', so if you have 0 and 1 as labels, it will 
# actually create a third, -1!!! So, I'll replace the 0-values in Y_test with -1
# Y_test[Y_test==0] = -1

from sklearn.grid_search import GridSearchCV
from sklearn.svm import OneClassSVM

param_grid = [{
     #'kernel': ['rbf'], 
     'gamma': [1e-1,1e-2,1e-3], 
     'nu': [.1, .3, .6, .9]}]

scores = ['recall', 'accuracy', 'precision', 'roc_auc']

for score in scores:
    print("# TUNING HYPER-PARAMETERS FOR %s" % score)
    print('---')

    svm_clf = GridSearchCV(OneClassSVM(), param_grid, cv=3, scoring=score, verbose=True)
    svm_clf.fit(X_train[Y_train==1], Y_train[Y_train==1])
#     for params, mean_score, scores in svm.grid_scores_:
#         print("%0.3f (+/-%0.03f) for %r"
#               % (mean_score, scores.std() / 2, params))
      
    print(classification_report(Y_test, svm_clf.predict(X_test)))       
# print 'Overall best One-Class-SVM'
# print classification_report(Y_test, svm_clf.best_estimator_.predict(X_test))


full_data.Supplier_Minimum_Distance_Investigated_Projects_Cumulative


# # HAVE TO MAKE SURE YOU'RE NOT USING THE FUTURE TO PREDICT THE PAST
# # Approach:
# #  1. Sort by date
# #  2. Set aside x% for test (the same set every time)
# #  3. K-fold the train set, and use only the training portion of it to train

# # the classifier!
# clf = svm.OneClassSVM(nu=0.1, kernel="rbf", gamma=0.1)
# # clf = svm.SVC
# #the classifier is defined above: svm

# X_for_k_fold = X[:-1000]
# X_test = X[-1000:]
# Y = array([1.0]*len(X))

# # fit the model
# kf = KFold(len(X_for_k_fold), n_folds=5, shuffle=True)

# recalls=[]
# for train_index, test_index in kf:
#     X_train = X_for_k_fold[train_index]
#     clf.fit(X_train)
# #     clf.fit(X_for_k_fold)
#     recall = sum(clf.predict(X_train)==1) / float(len(X_train))  # total predicted as 1 / length of test set (all 1's)
# #     print classification_report([1.]*len(X_test), clf.predict(X_test))
#     print 'Recall:', recall
#     recalls.append(recall)
# print 'Average recall:', mean(recalls)


X.shape


unique(np.isfinite(X_train))


# #Logistic Regression


X_train


from sklearn.linear_model import LogisticRegression
log_clf = LogisticRegression()
log_clf.fit(X_train, Y_train)
print classification_report(Y_test, log_clf.predict(X_test))


# FEATURES
print
print 'FEATURE IMPORTANCES'
print 'Odds ratio of intercept (percent_rejecte = total_bids = 0):', exp(logest.intercept_)
print 'Odds ratios of percent_rejected and total_bids:', exp(logest.coef_)






# #Two-class SVM


# Without tuning parameters
# --


svm_clf = svm.SVC()
svm_clf.fit(X_train, Y_train)
print classification_report(Y_test, svm_clf.predict(X_test))


# With grid search to tune parameters
# --


from sklearn.grid_search import GridSearchCV

param_grid = [
    {
     #'kernel': ['rbf'], 
     'gamma': [1e-1,1e-2],#,1e-3], 
     'C': [.1]#, 1, 10]# 100, 250]}
}]

scores = ['recall', 'accuracy', 'precision', 'roc_auc']

for score in scores:
    print("# TUNING HYPER-PARAMETERS FOR %s" % score)
    print('---')

    svm_clf = GridSearchCV(SVC(probability=True), param_grid, cv=3, scoring=score, verbose=True)
    svm_clf.fit(X_train, Y_train)    
    print("Best parameters set found on training set:")
    print(svm.best_estimator_)
    print('---')
    print("Grid scores on development set:")
    for params, mean_score, scores in svm.grid_scores_:
        print("%0.3f (+/-%0.03f) for %r"
              % (mean_score, scores.std() / 2, params))
    print('---')

print("Detailed classification report:")
print(classification_report(Y_test, svm_clf.best_estimator_.predict(X_test)))    
print(classification_report(Y_test, svm_clf.predict(X_test)))








# #How much info do we have for the 'Unfounded' class?


print 'How many unique companies are in overlap and "Unfounded"?'
print '"Unfounded" case:', overlap[overlap.outcome_of_overall_investigation_when_closed == 'Unfounded'].canonical_name.unique().shape[0]
print '"Unfounded" allegation:', overlap[overlap.allegation_outcome == 'Unfounded'].canonical_name.unique().shape[0]


# EVENTUALLY, THE OUTPUT I WANT IS A DATAFRAME IN WHICH
# canonical_name, source {overlap_train, overlap_test, contracts}, score {probability or ranking or whatever}



# #THREE EVALUATION OUTCOMES
# 
# 1. Predictions on three sets {overlap_train, overlap_test, and other procurements}
# --


y_pred_train






# print pd.crosstab([1]*len(y_pred_train), y_pred_train) #, rownames=['Actual'], colnames=['Predicted'])



from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

# predict
y_pred_train = clf.predict(df_overlap_train[features].dropna()).astype(float)
y_pred_test = clf.predict(df_overlap_test[features].dropna()).astype(float)
y_pred_random = clf.predict(df_nonoverlap_random[features].dropna()).astype(float)

print '\n\nCLASSIFICATION REPORTS\n----------------------------\n'
print 'On OVERLAP_TRAIN (66% of overlap data)'

print classification_report([1]*len(y_pred_train), y_pred_train)
print 'On OVERLAP_TEST (33% of overlap data)'
print classification_report([1]*len(y_pred_test), y_pred_test)
print 'On RANDOM NONOVERLAP PROCUREMENTS (2000 rows)'
print classification_report([1]*len(y_pred_random), y_pred_random)

print '\n\nCONFUSION MATRICES\n--------------------------------\n'
# define helper function which will use Pandas to print the matrices
def print_cm(cm, labels):
    print pd.DataFrame(cm, columns=labels, index=labels)
labels=['Non-invest', 'Invest']

print 'On OVERLAP_TRAIN (66% of overlap data)'
print print_cm(confusion_matrix([1]*len(y_pred_train), y_pred_train), labels)
# print pd.crosstab([1]*len(y_pred_train), y_pred_train, rownames=['Actual'], colnames=['Predicted'])
print 'On OVERLAP_TEST (33% of overlap data)'
print print_cm(confusion_matrix([1]*len(y_pred_test), y_pred_test), labels)
print 'On RANDOM NONOVERLAP PROCUREMENTS (2000 rows)'
print print_cm(confusion_matrix([1]*len(y_pred_random), y_pred_random), labels)
   


# Typical classifier evaluation metrics

# ROC curve and area the curve
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


# 2. Percent of guilty's found vs. risk score
# --


# from each classifier need probability or risk score
svm.predict_proba


# 3. Percent of guilty's found vs. percent of mixture (of guiltys/random procurements) examined 
# --









