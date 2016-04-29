#SVM
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

# RANDOM FOREST
param_grid = [
  {'n_estimators': [5,10,15,20]}
]

scores = ['accuracy','precision', 'recall', 'roc_auc']

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
