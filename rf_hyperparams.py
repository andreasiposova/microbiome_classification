import argparse
import os
import pandas as pd
import numpy as np

from utils import setup_logging, Config
from preprocessing import preprocess_data

from datetime import datetime

from visualization import sensitivity_plot, grid_search_plot, get_rf_scores_params, cm_plot

from data_loading import load_tsv_files

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, fbeta_score, confusion_matrix
from sklearn.model_selection import GridSearchCV

# Set up logging
#logfile = setup_logging("tune_random_forest") # logger

# Set up logging
#logger = setup_logging("tune_random_forest")
#log_file = "rf" + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

FUDAN = 'fudan'
HUADONG1 = 'huadong1'
HUADONG2 = 'huadong2'

file_names = list(("pielou_e_diversity", "simpson_diversity", "phylum_relative", "observed_otus_diversity", "family_relative",
"class_relative", "fb_ratio", "enterotype", "genus_relative", "species_relative", "shannon_diversity", "domain_relative",
"order_relative", "simpson_e_diversity"))

yang_metadata_path = "data/Yang_PRJNA763023/metadata.csv"
fudan_filepath = 'data/Yang_PRJNA763023/Yang_PRJNA763023_SE/parsed/normalized_results/'
huadong_filepath_1 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_1/parsed/normalized_results'
huadong_filepath_2 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_2/parsed/normalized_results'





def grid_search_rf(X_train, X_test, y_train, y_test, file_name):

    param_grid = {
        'n_estimators': [2],# 50, 100, 200],
        'max_depth': [5, 20], #, 5, 10, 20],
        'min_samples_split': [2, 4], #, 5, 10],
        'min_samples_leaf': [1, 4],# 2, 4],
        'max_features': ['auto', 'sqrt'],# 'sqrt', 'log2']
        'random_state': [1234]
    }

    # Define the scoring methods
    scoring = {
        'recall': make_scorer(recall_score),
    }

    # Instantiate the classifier
    rf = RandomForestClassifier()

    # Perform the grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring=scoring, refit='recall',
                               return_train_score=True, n_jobs=6)
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
    return test_scores, best_res, best_estimator_params, y_test, y_pred



def visualize_results(test_scores, data_name, file_name, y_test, y_pred):
    rf_scores, rf_param_combinations, n_estimators_mean_metrics, max_depth_mean_metrics, max_features_mean_metrics, min_samples_split_mean_metrics, min_samples_leaf_mean_metrics = get_rf_scores_params(test_scores)
    # Extract the parameter values
    grid_search_plot(rf_param_combinations, rf_scores, data_name, file_name)
    sensitivity_plot(n_estimators_mean_metrics, data_name, file_name)
    sensitivity_plot(max_depth_mean_metrics, data_name, file_name)
    sensitivity_plot(max_features_mean_metrics, data_name, file_name)
    sensitivity_plot(min_samples_split_mean_metrics, data_name, file_name)
    sensitivity_plot(min_samples_leaf_mean_metrics, data_name, file_name)
    cm_plot(y_test, y_pred, data_name, file_name)



def run_rf_tuning(data_name, filepath):
    full_results = []
    data = load_tsv_files(filepath)
    for key in data:
        X_train, X_test, y_train, y_test = preprocess_data(data[key], yang_metadata_path) #preprocess_fudan_data?
        scores, best_results, best_estimator, y_test, y_pred = grid_search_rf(X_train, X_test, y_train, y_test, file_name=key)
        full_results.append([str(key),best_estimator, best_results])
        visualize_results(scores, data_name, key, y_test, y_pred)
        print(full_results)

    results_table = pd.DataFrame()
    df = pd.DataFrame(columns=['Normalization'])
    for i in full_results:
        # Create a dataframe with 'Normalization' as the first column
        df.loc[0] = i[0]
        # Iterate through the rest of the list and add the key-value pairs as columns and rows
        for d in i[1:]:
            for k, value in d.items():
                if k not in df.columns:
                    df[k] = None
                df.loc[0, k] = value
        results_table = results_table.append(df)
        results_table.drop(['oob_score', 'min_weight_fraction_leaf', 'bootstrap', 'ccp_alpha', 'class_weight', 'min_impurity_decrease', 'min_impurity_split'], inplace=True, axis=1)


    latex_table = results_table.to_latex()

    if not os.path.exists(str(Config.LOG_DIR) + "/" + str(data_name)):
        os.makedirs(os.path.join(Config.LOG_DIR, data_name))

    with open(os.path.join(Config.LOG_DIR, data_name, f"best_results.txt"), "w") as f:
        f.write(latex_table)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=HUADONG1)
    parser.add_argument('--filepath', type=str, default=huadong_filepath_1)

    args = parser.parse_args()
    data_name = args.data_name
    if data_name == FUDAN:
        run_rf_tuning(data_name=args.data_name, filepath=args.filepath)
    elif data_name == HUADONG1:
        run_rf_tuning(data_name=args.data_name, filepath=args.filepath)
    else:
        raise ValueError()


