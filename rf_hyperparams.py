import argparse
import os
import pandas as pd
import numpy as np

from utils import setup_logging, Config
from preprocessing import preprocess_data, preprocess_huadong

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

file_names = list(("pielou_e_diversity", "simpson_diversity", "phylum_relative", "observed_otus_diversity", "family_relative", "class_relative", "fb_ratio", "enterotype", "genus_relative", "species_relative", "shannon_diversity", "domain_relative", "order_relative", "simpson_e_diversity"))

yang_metadata_path = "data/Yang_PRJNA763023/metadata.csv"
fudan_filepath = 'data/Yang_PRJNA763023/Yang_PRJNA763023_SE/parsed/normalized_results/'
huadong_filepath_1 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_1/parsed/normalized_results'
huadong_filepath_2 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_2/parsed/normalized_results'





def grid_search_rf(X_train, X_test, y_train, y_test, X_val, y_val, file_name):

    param_grid = {
        'n_estimators': np.linspace(20, 800, 50, dtype=int),
        'max_depth': np.linspace(2, 40, 3, dtype=int),
        'min_samples_split': np.linspace(2, 50, 3, dtype=int),
        'min_samples_leaf': np.linspace(2, 50, 3, dtype=int),
        'max_features': ['auto', 'sqrt', 'log2'], # 'sqrt', 'log2']
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
                               return_train_score=True, n_jobs=-1)
    print(f"fitting GridSearch on {file_name}")
    grid_search.fit(X_train, y_train)

    # Get the best estimator
    best_estimator = grid_search.best_estimator_
    results = grid_search.cv_results_
    test_scores = []
    #for every combination of params, fit the model with those params to train set
    #then evaluate on test set
    #get the best model based on the test set
    # later get that model and eval on validation set
    for i in range(len(grid_search.cv_results_['params'])):
        rf_ = RandomForestClassifier(**grid_search.cv_results_['params'][i])
        rf_.fit(X_train, y_train)
        y_pred = rf_.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        roc_auc = roc_auc_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)
        f2 = fbeta_score(y_test, y_pred, beta=2)
        #these test scores are the all param combinations from the cv above evaluated on the test set to choose the best params
        #because the best estimator returned from gridsearch is only best on the train data (cv introduces some data leakage maybe?)

        test_scores.append({'file': file_name, 'params': grid_search.cv_results_['params'][i],
                            'accuracy': accuracy, 'precision': precision,
                            'recall': recall, 'roc_auc': roc_auc, 'f1': f1, 'f2': f2})

    #for each file, get the metrics on the test set
    # for each combination of params
    # choose the one with the highest roc_auc
    # use that best model to evaluate on the huadong val set

    # find the test score respective to the highest roc_auc for the corresponding data file
    max_roc_auc = max(test_scores, key=lambda x: x['roc_auc'])
    #get the params corresponding to the highest roc_auc on the test set (30% of the fudan cohort)
    max_params = max_roc_auc['params']

    val_scores = []
    #apply the best model params (chosen on the test set)
    #predict on the validation set (huadong cohort)
    rf_best_test = RandomForestClassifier(**max_params)
    rf_best_test.fit(X_train, y_train)
    y_val_pred = rf_best_test.predict(X_val)
    #compute the metrics on the validation set
    accuracy = accuracy_score(y_val, y_val_pred)
    precision = precision_score(y_val, y_val_pred)
    recall = recall_score(y_val, y_val_pred)
    roc_auc = roc_auc_score(y_val, y_val_pred)
    f1 = f1_score(y_val, y_val_pred)
    f2 = fbeta_score(y_val, y_val_pred, beta=2)
    #get all the params and scores on the validation set as a list (e.g.filename + best params found on the test set + validation score)
    val_scores.append({'file': file_name, 'params': max_params,
                            'accuracy': accuracy, 'precision': precision,
                            'recall': recall, 'roc_auc': roc_auc, 'f1': f1, 'f2': f2})
    best_val_eval = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'roc_auc': roc_auc, 'f1': f1, 'f2': f2}

    # Get the best estimator
    #best estimator found by the gridsearch on train set only
    best_estimator_params_on_train = best_estimator.get_params()
    rf_best_train = RandomForestClassifier(**best_estimator_params_on_train)
    # Evaluate the best estimator on X_test and y_test
    rf_best_train.fit(X_train, y_train)
    y_pred = rf_best_train.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    f2 = fbeta_score(y_test, y_pred, beta=2)

    best_train_est_test_eval = {'accuracy': accuracy, 'precision': precision, 'recall': recall, 'roc_auc': roc_auc, 'f1': f1, 'f2': f2}

    # Return the results
    return test_scores, val_scores, best_train_est_test_eval, best_estimator_params_on_train, best_val_eval, max_params, y_test, y_pred, y_val_pred



def visualize_results(test_scores, data_name, file_name, y_test, y_pred, y_val, y_pred_val):
    rf_scores, rf_param_combinations, n_estimators_mean_metrics, max_depth_mean_metrics, max_features_mean_metrics, min_samples_split_mean_metrics, min_samples_leaf_mean_metrics = get_rf_scores_params(test_scores)
    # Extract the parameter values
    grid_search_plot(rf_param_combinations, rf_scores, data_name, file_name)
    sensitivity_plot(n_estimators_mean_metrics, data_name, file_name)
    sensitivity_plot(max_depth_mean_metrics, data_name, file_name)
    sensitivity_plot(max_features_mean_metrics, data_name, file_name)
    sensitivity_plot(min_samples_split_mean_metrics, data_name, file_name)
    sensitivity_plot(min_samples_leaf_mean_metrics, data_name, file_name)
    cm_plot(y_test, y_pred, data_name, file_name)
    cm_plot(y_val, y_pred_val, data_name, file_name)

def create_results_table(full_results):
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
    return df

def save_result_table(results_table, data_name, table_name):
    latex_table = results_table.to_latex()

    if not os.path.exists(str(Config.LOG_DIR) + "/" + str(data_name)):
        os.makedirs(os.path.join(Config.LOG_DIR, data_name))

    with open(os.path.join(Config.LOG_DIR, data_name, f"{table_name}.txt"), "w") as f:
        f.write(latex_table)

    results_table.to_csv(os.path.join(Config.LOG_DIR, data_name, f"{table_name}.csv"))


def run_rf_tuning(data_name, filepath, h1_filepath, h2_filepath):
    full_results_test = []
    full_results_val = []
    results_test_table = pd.DataFrame()
    results_val_table = pd.DataFrame()
    data = load_tsv_files(filepath)
    huadong_data1 = load_tsv_files(huadong_filepath_1)
    huadong_data2 = load_tsv_files(huadong_filepath_2)
    for key in data:
        X_train, X_test, y_train, y_test = preprocess_data(data[key], yang_metadata_path) #preprocess_fudan_data?
        X_h1, y_h1 = preprocess_huadong(huadong_data1[key], yang_metadata_path)
        X_h2, y_h2 = preprocess_huadong(huadong_data2[key], yang_metadata_path)
        X_val = pd.concat([X_h1, X_h2])
        y_val = y_h1 + y_h2
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


        scores, val_scores, best_results_on_train, best_estimator, best_results_on_val, best_params_on_test, y_test, y_pred, y_val_pred = grid_search_rf(X_train, X_test, y_train, y_test, X_val, y_val, file_name=key)
        full_results_test.append([str(key),best_estimator, best_results_on_train])
        full_results_val.append([str(key),best_params_on_test, best_results_on_val])
        visualize_results(scores, data_name, key, y_test, y_pred, y_val, y_val_pred)

        best_results_test = create_results_table(full_results_test)
        best_results_val = create_results_table(full_results_val)
        untuned_params = ['oob_score', 'min_weight_fraction_leaf', 'bootstrap',
                            'ccp_alpha', 'class_weight', 'min_impurity_decrease',
                            'min_impurity_split', 'max_leaf_nodes', 'max_samples',
                            'verbose', 'warm_start']
        results_test_table = results_test_table.append(best_results_test)
        results_val_table = results_test_table.append(best_results_val)
        results_test_table.drop(untuned_params, inplace=True, axis=1)
        results_val_table.drop(untuned_params, inplace=True, axis=1)
        save_result_table(results_test_table, data_name, table_name="best_results_test")
        save_result_table(results_val_table, data_name, table_name="best_results_val")



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=FUDAN)
    parser.add_argument('--filepath', type=str, default=fudan_filepath)
    parser.add_argument('--h1_filepath', type=str, default=huadong_filepath_1)
    parser.add_argument('--h2_filepath', type=str, default=huadong_filepath_2)

    args = parser.parse_args()
    data_name = args.data_name
    if data_name == FUDAN:
        run_rf_tuning(data_name=args.data_name, filepath=args.filepath, h1_filepath=args.h1_filepath, h2_filepath=args.h2_filepath)
    elif data_name == HUADONG1:
        run_rf_tuning(data_name=args.data_name, filepath=args.filepath, h1_filepath=args.h1_filepath, h2_filepath=args.h2_filepath)
    else:
        raise ValueError()


