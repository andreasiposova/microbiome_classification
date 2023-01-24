from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

def calculate_feature_importance(X, y):
    # initialize the random forest model
    clf = RandomForestClassifier(random_state=1234, n_jobs=-1)

    # initialize the k-fold cross-validation
    kf = KFold(n_splits=10, random_state=1234)

    # initialize the feature importance dictionary
    feature_importance = {}

    # iterate over each feature
    #for i in range(X.shape[1]):
        # create a new dataset with one feature removed
        #X_new = X[:, [j for j in range(X.shape[1]) if j != i]]
    for feature in X.columns:
        # create a new dataset by excluding the current feature
        X_new = X[[col for col in X.columns if col != feature]]
        # add the new dataset to the list of new datasets
        # perform k-fold cross validation
        scores = cross_val_score(clf, X_new, y, cv=kf, n_jobs = -1, random_state=1234)
        # calculate the mean decrease in accuracy
        mean_decrease_accuracy = 1 - scores.mean()
        # add the feature and its importance to the dictionary
        feature_importance.update({feature: mean_decrease_accuracy})

    # sort the features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    # print the sorted features
    for feature, importance in sorted_features:
        print(f"{feature}: {importance:.4f}")


    # print the top 40 most important features
    top_features = sorted_features[:40]

    # Return the top 40 most important features
    return top_features
