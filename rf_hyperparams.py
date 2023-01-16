import os

import pandas as pd
from utils import setup_logging, Config
from preprocessing import preprocess_data
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.pipeline import Pipeline
import csv
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.model_selection import GridSearchCV
from datetime import datetime
import matplotlib.pyplot as plt
from visualization import sensitivity_plot, get_scores

from skopt import BayesSearchCV
from skopt.space import Integer, Categorical
from data_loading import load_data, load_tsv_files

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, \
    fbeta_score
from sklearn.model_selection import GridSearchCV

# Set up logging
#logfile = setup_logging("tune_random_forest") # logger

# Set up logging
logger = setup_logging("tune_random_forest")
log_file = "rf" + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")



file_names = list(("pielou_e_diversity", "simpson_diversity", "phylum_relative", "observed_otus_diversity", "family_relative",
"class_relative", "fb_ratio", "enterotype", "genus_relative", "species_relative", "shannon_diversity", "domain_relative",
"order_relative", "simpson_e_diversity"))

yang_metadata_path = "data/Yang_PRJNA763023/metadata.csv"
fudan_filepath = 'data/Yang_PRJNA763023/Yang_PRJNA763023_SE/parsed/normalized_results/'

fudan_data = load_tsv_files(fudan_filepath)





def grid_search_rf(X_train, X_test, y_train, y_test, file_name):

    param_grid = {
        'n_estimators': [5, 50, 100],# 50, 100, 200],
        'max_depth': [None], #, 5, 10, 20],
        'min_samples_split': [2], #, 5, 10],
        'min_samples_leaf': [1],# 2, 4],
        'max_features': ['auto'],# 'sqrt', 'log2']
    }

    # Define the scoring methods
    scoring = {
        'recall': make_scorer(recall_score),
    }

    # Instantiate the classifier
    rf = RandomForestClassifier()

    # Perform the grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring=scoring, refit='recall',
                               return_train_score=True)
    grid_search.fit(X_train, y_train)

    # Get the best estimator
    best_estimator = grid_search.best_estimator_
    results = grid_search.cv_results_
    test_scores = []

    for i in range(len(grid_search.cv_results_['params'])):
        print(grid_search.cv_results_['params'][i])
        rf_ = RandomForestClassifier(**grid_search.cv_results_['params'][i])
        rf_.fit(X_train, y_train)
        y_pred = rf_.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)

        test_scores.append({'file': file_name, 'params': grid_search.cv_results_['params'][i],
                            'accuracy': accuracy, 'precision': precision,
                            'recall': recall, 'roc_auc': roc_auc, 'f1': f1, 'f2': f2})

    print(test_scores)

    # Get the best estimator
    best_estimator_params = best_estimator.get_params()

    # Evaluate the best estimator on X_test and y_test
    y_pred = best_estimator.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)
    best_res = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'roc_auc': roc_auc, 'f1': f1, 'f2': f2}
    # Return the results
    return test_scores, best_res, best_estimator_params



def visualize_results(test_scores, file_name):
    # Create a folder named "plots" if it doesn't exist
    if not os.path.exists('plots'):
        os.makedirs('plots')
    if not os.path.exists('plots/' + str(file_name)):
        os.makedirs('plots/' + str(file_name))

    scores = get_scores(test_scores)

    # Extract the parameter values
    param_values = [score['params'] for score in test_scores]
    n_estimators = [param['n_estimators'] for param in param_values]
    max_depth = [param['max_depth'] for param in param_values]
    min_samples_split = [param['min_samples_split'] for param in param_values]
    min_samples_leaf = [param['min_samples_leaf'] for param in param_values]
    max_features = [param['max_features'] for param in param_values]

    sensitivity_plot(n_estimators, scores, file_name, "n_estimators")
    sensitivity_plot(max_depth, scores, file_name, "max_depth")
    sensitivity_plot(min_samples_split, scores, file_name, "min_samples_split")
    sensitivity_plot(min_samples_leaf, scores, file_name, "min_samples_leaf")
    sensitivity_plot(max_features, scores, file_name, "max_features")

    # Plot the relationship between accuracy and n_estimators

    #plt.savefig('images/' + file_name)


full_results = []
for key in fudan_data:
    X_train, X_test, y_train, y_test = preprocess_data(fudan_data[key], yang_metadata_path)
    scores, best_results, best_estimator = grid_search_rf(X_train, X_test, y_train, y_test, file_name=key)
    full_results.append([str(key),best_estimator, best_results])
    visualize_results(scores, key)
    print(full_results)


d = {'Diversity': [], 'best parameters': [], 'metrics': []}
for i in full_results:
    d['Diversity'].append(i[0])
    d['best parameters'].append(i[1])
    d['metrics'].append(i[2])

df = pd.DataFrame(d)
df.set_index(['Diversity'], inplace=True)

# split 'metrics' column into separate columns
df_metrics = pd.json_normalize(df['metrics'])

# join the new DataFrame with the original one
df = df.join(df_metrics)

# drop the 'metrics' column
df.drop(['metrics'], axis=1, inplace=True)

latex_table = df.to_latex()

with open(os.path.join(Config.LOG_DIR, f"fudan/best_results.txt"), "w") as f:
    f.write(latex_table)





