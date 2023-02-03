import argparse
import os
import pandas as pd
import numpy as np
from statistics import mean
import os
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils import Config

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

from feature_selection import calculate_feature_importance, select_features_from_paper
from utils import setup_logging, Config
from preprocessing import preprocess_data, preprocess_huadong, load_young_old_labels, preprocess_with_y_o_labels, \
    preprocess_huadong_with_y_o_labels, full_preprocessing_y_o_labels

from datetime import datetime

from visualization import sensitivity_plot, grid_search_plot, get_rf_scores_params, cm_plot, grid_search_train_test_plot

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

file_names = list(("pielou_e_diversity", "simpson_diversity", "phylum_relative", "observed_otus_diversity", "family_relative", "class_relative", "fb_ratio", "enterotype", "genus_relative", "species_relative", "shannon_diversity", "domain_relative", "order_relative", "simpson_e_diversity"))

yang_metadata_path = "data/Yang_PRJNA763023/metadata.csv"
fudan_filepath = 'data/Yang_PRJNA763023/Yang_PRJNA763023_SE/parsed/normalized_results/'
huadong_filepath_1 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_1/parsed/normalized_results'
huadong_filepath_2 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_2/parsed/normalized_results'
young_old_labels_path = 'data/Yang_PRJNA763023/SraRunTable.csv'





def grid_search_rf(X_train, X_test, y_train, y_test, X_val, y_val, data_name, file_name, group):
    if group == "old":
        param_grid = {
            'n_estimators': np.arange(5, 50, 2, dtype=int),
            'max_depth': np.arange(2, 12, 1, dtype=int),
            'min_samples_split': np.arange(2, 15, 2, dtype=int),
            'min_samples_leaf': np.arange(2, 30, 2, dtype=int),
            'max_features': ['sqrt', 'log2'], # 'sqrt', 'log2'],
            #'criteria': ['gini', 'entropy'],
            'random_state': [1234],
            'class_weight': ["balanced_subsample"]
        }
    if group == "young":
        param_grid = {
            'n_estimators': np.arange(10, 400, 10, dtype=int),
            'max_depth': np.arange(2, 20, 1, dtype=int),
            'min_samples_split': np.arange(2, 20, 2, dtype=int),
            'min_samples_leaf': np.arange(2, 20, 2, dtype=int),
            'max_features': ['sqrt', 'log2'],  # 'sqrt', 'log2'],
            'random_state': [1234],
            'class_weight': ["balanced_subsample"]
        }

    if group == "all":
        param_grid = {
            'n_estimators': np.arange(10, 1100, 20, dtype=int),
            'max_depth': np.arange(2, 20, 10, dtype=int),
            'min_samples_split': np.arange(2, 30, 10, dtype=int),
            'min_samples_leaf': np.arange(2, 30, 10, dtype=int),
            'max_features': ['sqrt', 'log2'],  # 'sqrt', 'log2'],
            'random_state': [1234],
            'class_weight': ['balanced_subsample']
        }

    # Define the scoring methods
    scoring = {
        'roc_auc': make_scorer(roc_auc_score),
        'precision': make_scorer(accuracy_score),
        #'f1': make_scorer(f1_score)
    }

    # Instantiate the classifier
    rf = RandomForestClassifier()
    train_scores_gridsearch = []
    test_scores_gridsearch = []

    # Perform the grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=10, scoring=scoring, refit='roc_auc',
                               return_train_score=True, n_jobs=-1)
    print(f"fitting GridSearch on {file_name}")
    grid_search.fit(X_train, y_train)

    # Get the best estimator
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    roc_auc_grid = grid_search.best_score_
    results = grid_search.cv_results_


    train_scores_gridsearch = results.get('mean_train_roc_auc')
    test_scores_gridsearch = results.get('mean_test_roc_auc')



    # extract the train and test scores from the grid search
    #train_scores_gridsearch = [grid_search.score(X_train, y_train) for _ in range(len(grid_search.cv_results_['params']))]
    #test_scores_gridsearch = [grid_search.score(X_test, y_test) for _ in range(len(grid_search.cv_results_['params']))]
    grid_search_train_test_plot(train_scores_gridsearch, test_scores_gridsearch, data_name, group)


    test_scores = []


    #for every combination of params, fit the model with those params to train set
    #then evaluate on test set
    #get the best model based on the test set
    # later get that model and eval on validation set
    grid_val_scores = []
    for i in range(len(grid_search.cv_results_['params'])):
        rf_ = RandomForestClassifier(**grid_search.cv_results_['params'][i])
        rf_.fit(X_train, y_train)
        y_pred = rf_.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred)
        f1 = f1_score(y_val, y_pred)
        f2 = fbeta_score(y_val, y_pred, beta=2)

        #these test scores are the all param combinations from the cv above evaluated on the test set to choose the best params
        #because the best estimator returned from gridsearch is only best on the train data (cv introduces some data leakage maybe?)

        grid_val_scores.append({'file': file_name, 'params': grid_search.cv_results_['params'][i],
                            'accuracy': accuracy, 'precision': precision,
                            'recall': recall, 'roc_auc': roc_auc, 'f1': f1, 'f2': f2})

    #for each file, get the metrics on the test set
    # for each combination of params
    # choose the one with the highest roc_auc
    # use that best model to evaluate on the huadong val set

    # find the test score respective to the highest roc_auc for the corresponding data file
    #max_roc_auc = max(grid_val_scores, key=lambda x: x['roc_auc'])
    #for i in range(len(grid_val_scores)):
     #   keys = [k for k, v in grid_val_scores[i].items() if 0.8 <= v['roc_auc'] <= 0.9]

    results_auroc = [d for d in grid_val_scores if 0.78 <= d['roc_auc'] <= 1.0]
    best_auroc_params = max(results_auroc, key= lambda x: x['roc_auc'])
    best_params = best_auroc_params['params']

    #get the params corresponding to the highest roc_auc on the test set (30% of the fudan cohort)
    #max_params = max_roc_auc['params']
    print("params with max roc_auc: ", results_auroc)
    # Get the best estimator
    #best estimator found by the gridsearch on train set only

    best_estimator_params_on_train = best_estimator.get_params()
    train_scores = []

    rf_best_train = RandomForestClassifier(**best_params)
    # Evaluate the best estimator on X_test and y_test
    rf_best_train.fit(X_train, y_train)
    y_pred_train = rf_best_train.predict(X_train)
    accuracy = accuracy_score(y_train, y_pred_train)
    precision = precision_score(y_train, y_pred_train)
    recall = recall_score(y_train, y_pred_train)
    roc_auc = roc_auc_score(y_train, y_pred_train)
    f1 = f1_score(y_train, y_pred_train)
    f2 = fbeta_score(y_train, y_pred_train, beta=2)

    train_scores.append({'file': file_name, 'params': best_params,
                       'accuracy': accuracy, 'precision': precision,
                       'recall': recall, 'roc_auc': roc_auc, 'f1': f1, 'f2': f2})

    best_train_set_res = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'roc_auc': roc_auc, 'f1': f1,
                         'f2': f2}

    val_scores = []
    #apply the best model params (chosen on the test set)
    #predict on the validation set (huadong cohort)
    #rf_best_test = RandomForestClassifier(**max_params)
    #rf_best_test.fit(X_train, y_train)
    #y_val_pred = rf_best_test.predict(X_val)
    best_estimator_params_on_train = best_estimator.get_params()

    # Evaluate the best estimator on X_val and y_val - HUADONG Cohort
    rf_best_train.fit(X_train, y_train)
    y_val_pred = rf_best_train.predict(X_val)

    #compute the metrics on the validation set
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    f2 = fbeta_score(y_val, y_val_pred, beta=2)
    #get all the params and scores on the validation set as a list (e.g.filename + best params found on the test set + validation score)
    val_scores.append({'file': file_name, 'params': best_params,
                            'accuracy': accuracy, 'precision': precision,
                            'recall': recall, 'roc_auc': roc_auc, 'f1': f1, 'f2': f2})
    best_val_eval = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'roc_auc': roc_auc, 'f1': f1, 'f2': f2}

    # Get the best estimator
    #best estimator found by the gridsearch on train set only
    best_estimator_params_on_train = best_estimator.get_params()
    rf_best_train = RandomForestClassifier(**best_params)
    # Evaluate the best estimator on X_test and y_test
    rf_best_train.fit(X_train, y_train)
    y_pred = rf_best_train.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)

    test_scores.append({'file': file_name, 'params': best_params,
                       'accuracy': accuracy, 'precision': precision,
                       'recall': recall, 'roc_auc': roc_auc, 'f1': f1, 'f2': f2})

    best_test_set_res = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'roc_auc': roc_auc, 'f1': f1, 'f2': f2}



    # Return the results
    return train_scores, test_scores, val_scores, best_train_set_res, best_test_set_res, best_params, best_auroc_params, best_val_eval, y_pred_train, y_pred, y_val_pred



def visualize_results(test_scores, data_name, group, file_name, y_train, y_train_pred, y_test, y_pred, y_val, y_pred_val):
    #rf_scores, rf_param_combinations, n_estimators_mean_metrics, max_depth_mean_metrics, max_features_mean_metrics, min_samples_split_mean_metrics, min_samples_leaf_mean_metrics, class_weight_mean_metrics  = get_rf_scores_params(test_scores)
    # Extract the parameter values
    #grid_search_plot(rf_param_combinations, rf_scores, data_name, file_name)
    #sensitivity_plot(n_estimators_mean_metrics, data_name, file_name)
    #sensitivity_plot(max_depth_mean_metrics, data_name, file_name)
    #sensitivity_plot(max_features_mean_metrics, data_name, file_name)
    #sensitivity_plot(min_samples_split_mean_metrics, data_name, file_name)
    #sensitivity_plot(min_samples_leaf_mean_metrics, data_name, file_name)
    #sensitivity_plot(class_weight_mean_metrics, data_name, file_name)
    cm_plot(y_train, y_train_pred, data_name, group, file_name, "test")
    cm_plot(y_test, y_pred, data_name, group, file_name, "test")
    cm_plot(y_val, y_pred_val, data_name, group, file_name, "val")

def create_results_table(full_results):
    df = pd.DataFrame(columns=['Group'])
    for i in full_results:
        # Create a dataframe with 'Group' as the first column
        df.loc[0] = i[0]
        #df.loc[1] = i[1]
        # Iterate through the rest of the list and add the key-value pairs as columns and rows
        for d in i[1:]:
            for k, value in d.items():
                if k not in df.columns:
                    df[k] = None
                df.loc[0, k] = value
    return df

def save_result_table(results_train_table, results_test_table, results_val_table, data_name, file_name, group, table_name):
    results_table = pd.DataFrame()
    mytable = [results_train_table, results_test_table, results_val_table]
    results_table = pd.concat(mytable)
    #results_table = results_table.append(results_train_table, ignore_index=True)
    #results_table = results_table.append(results_test_table, ignore_index=True)
    #results_table = results_table.append(results_val_table, ignore_index=True)
    #results_table['Set'] = ['Train', 'Test', 'Validation']
    #results_table = results_table.pop('Set')

    latex_table = results_table.to_latex()


    if not os.path.exists(str(Config.LOG_DIR) + "/" + str(data_name) + "/" + group + "/" + str(file_name)):
        os.makedirs(os.path.join(Config.LOG_DIR, data_name, group, file_name))

    with open(os.path.join(Config.LOG_DIR, data_name, group, file_name, f"{table_name}.txt"), "w") as f:
        f.write(latex_table)

    results_table.to_csv(os.path.join(Config.LOG_DIR, data_name, group, file_name,  f"{table_name}.csv"))


def run_rf_tuning(data_name, filepath, group, select_features = True):
    full_results_train = []
    full_results_test = []
    full_results_val = []
    results_train_table = pd.DataFrame()
    results_test_table = pd.DataFrame()
    results_val_table = pd.DataFrame()

    y_o_labels = load_young_old_labels(young_old_labels_path)
    data = load_tsv_files(filepath)
    huadong_data1 = load_tsv_files(huadong_filepath_1)
    huadong_data2 = load_tsv_files(huadong_filepath_2)

    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    X_val = pd.DataFrame()

    for key in data:
        if key == "genus_relative" or key == "family_relative":
            #X_train, X_test, y_train, y_test = preprocess_data(data[key], yang_metadata_path) #preprocess_fudan_data?
            if group == "all":
                # get train, test, val sets from the necessary files (ALL samples, both young-onset or old-onset CRC vs healthy samples)
                X_train_1, X_test_1, y_train, y_test = preprocess_data(data[key], yang_metadata_path)
                X_h1, y_h1 = preprocess_huadong(huadong_data1[key], yang_metadata_path)
                X_h2, y_h2 = preprocess_huadong(huadong_data2[key], yang_metadata_path)
            else:
                #get train, test, val sets from the necessary files and chosen group (young-onset or old-onset CRC vs healthy samples)
                X_train_1, X_test_1, X_val_1, y_train, y_test, y_val = full_preprocessing_y_o_labels(data, huadong_data1, huadong_data2, key, yang_metadata_path, young_old_labels_path, group)
            #if we want to use all features, set the file_name
            if select_features == False:
                file_name = "all_features"
                print("All features will be used")
            #if we want to select features used in the paper, set the file_name and perform feature selection
            if select_features == True:
                file_name = "selected_features"
                print(f"Selecting features...")
                #feature selection by RF, 10-fold CV, excluding each feature at a time, based on mean decrease accuracy per feature
                # top_features = calculate_feature_importance(X_train, y_train, group)
                # top_features_names = list(map(lambda x: x[0], top_features))
                # print(top_features_names)
                # X_train = X_train[top_features_names]
                # X_train.to_csv('data/selected_features_old.csv')
                # common_cols_f = set(X_test.columns).intersection(X_train.columns)
                # common_cols_fv = set(X_val.columns).intersection(X_train.columns)
                # X_test = X_test[common_cols_f]
                # X_val = X_val[common_cols_fv]
                #feature selection based on the feature names from the Yang et.al. paper
                X_test_1 = select_features_from_paper(X_test_1, group, key)
                X_train_1 = select_features_from_paper(X_train_1, group, key)
                X_val_1 = select_features_from_paper(X_val_1, group, key)

            X_train = pd.concat([X_train, X_train_1], axis=1)
            X_test = pd.concat([X_test, X_test_1], axis=1)
            X_val = pd.concat([X_val, X_val_1], axis=1)



    common_cols_t = set(X_test.columns).intersection(X_val.columns)
    common_cols_v = set(X_val.columns).intersection(X_test.columns)

    #filling missing values in huadong cohort with zeros
    #as two files are concatenated for huadong cohort files
    #they contain columns that are not compatible
    #thus creating missing values - they are replaced with 0 as it means the abundace of that bacteria is anyway 0
    X_val = X_val.fillna(0)
    X_val = X_val[common_cols_v]
    X_train = X_train[common_cols_t]
    X_test = X_test[common_cols_t]
    #X_train = X_train.append(X_test)
    #y_train = y_train + y_test
    print("number of samples in training set: ", len(X_train))
    print("number of samples in test set: ", len(X_test))
    print("number of samples in validation set: ", len(X_val))

    print(f"Running experiments on {group} samples")

    #scaler = MinMaxScaler()
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    #X_val = scaler.transform(X_val)

    train_scores, scores, val_scores, best_results_train, best_results_test, best_estimator, best_auroc_params, best_results_on_val, y_train_pred, y_pred, y_val_pred = grid_search_rf(X_train, X_test, y_train, y_test, X_val, y_val, data_name, file_name, group)
    if select_features == True:
        full_results_train.append(['selected_features', best_estimator, best_results_train])
        full_results_test.append(['selected_features', best_estimator, best_results_test])
        full_results_val.append(['selected_features', best_estimator, best_results_on_val])
        visualize_results(scores, data_name, 'selected_features', group, y_train, y_train_pred, y_test, y_pred, y_val, y_val_pred)

    if select_features == False:
        full_results_train.append(['all_features', best_estimator, best_results_train])
        full_results_test.append(['all_features', best_estimator, best_results_test])
        full_results_val.append(['all_features', best_estimator, best_results_on_val])
        visualize_results(scores, data_name, 'all_features', group, y_train, y_train_pred, y_test, y_pred,
                          y_val, y_val_pred)

    if not os.path.exists(str(Config.LOG_DIR) + "/" + str(data_name) + "/" + group + "/" + str(file_name)):
        os.makedirs(os.path.join(Config.LOG_DIR, data_name, group, file_name))
    with open(os.path.join(Config.LOG_DIR, data_name, group, file_name, f"RF_best_params.txt"), "w") as f:
        f.write(str(best_auroc_params))

    best_results_train = create_results_table(full_results_train)
    best_results_test = create_results_table(full_results_test)
    best_results_val = create_results_table(full_results_val)
    untuned_params = ['oob_score', 'min_weight_fraction_leaf', 'bootstrap',
                        'ccp_alpha', 'class_weight', 'min_impurity_decrease',
                        'min_impurity_split', 'max_leaf_nodes', 'max_samples',
                        'verbose', 'warm_start']
    results_train_table = results_train_table.append(best_results_train)
    results_test_table = results_test_table.append(best_results_test)
    results_val_table = results_val_table.append(best_results_val)
#   results_test_table.drop(untuned_params, inplace=True, axis=1)
#   results_val_table.drop(untuned_params, inplace=True, axis=1)
    save_result_table(results_train_table, results_test_table, results_val_table, data_name, file_name, group, table_name="RF_best_results")
    #save_result_table(results_test_table, data_name, file_name, group, table_name="best_results_test")
    #save_result_table(results_val_table, data_name, file_name, group, table_name="best_results_val")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=FUDAN)
    parser.add_argument('--filepath', type=str, default=fudan_filepath)
    parser.add_argument('--group', type=str, default='old')
    parser.add_argument('--select_features', type=bool, default=True)

    args = parser.parse_args()
    data_name = args.data_name
    if data_name == FUDAN:
        run_rf_tuning(data_name=args.data_name, filepath=args.filepath, group="old", select_features=True)
    elif data_name == HUADONG1:
        run_rf_tuning(data_name=args.data_name, filepath=args.filepath, h1_filepath=args.h1_filepath, h2_filepath=args.h2_filepath)
    else:
        raise ValueError()


