import argparse
import os
import json
import pandas as pd

from utils import setup_logging, Config
from preprocessing import preprocess_data

from datetime import datetime

from visualization import sensitivity_plot, get_scores

from data_loading import load_tsv_files

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, \
    fbeta_score
from sklearn.model_selection import GridSearchCV

# Set up logging
#logfile = setup_logging("tune_random_forest") # logger

# Set up logging
logger = setup_logging("tune_random_forest")
log_file = "rf" + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

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
        'n_estimators': [5],# 50, 100, 200],
        'max_depth': [None], #, 5, 10, 20],
        'min_samples_split': [2, 4], #, 5, 10],
        'min_samples_leaf': [4],# 2, 4],
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
    return test_scores, best_res, best_estimator_params



def visualize_results(test_scores, data_name, file_name):
    # Create a folder named "plots" if it doesn't exist
    #if not os.path.exists('plots'):
    #    os.makedirs('plots')
    #if not os.path.exists('plots/' + data_name + str(file_name)):
    #    os.makedirs('plots/' + data_name + str(file_name))

    scores = get_scores(test_scores)

    # Extract the parameter values
    param_values = [score['params'] for score in test_scores]
    n_estimators = [param['n_estimators'] for param in param_values]
    max_depth = [param['max_depth'] for param in param_values]
    min_samples_split = [param['min_samples_split'] for param in param_values]
    min_samples_leaf = [param['min_samples_leaf'] for param in param_values]
    max_features = [param['max_features'] for param in param_values]

    sensitivity_plot(n_estimators, scores, data_name, file_name, "n_estimators")
    sensitivity_plot(max_depth, scores, data_name, file_name, "max_depth")
    sensitivity_plot(min_samples_split, scores, data_name, file_name, "min_samples_split")
    sensitivity_plot(min_samples_leaf, scores, data_name, file_name, "min_samples_leaf")
    sensitivity_plot(max_features, scores, data_name, file_name, "max_features")

    # Plot the relationship between accuracy and n_estimators

    #plt.savefig('images/' + file_name)

def run_rf_tuning(data_name, filepath):
    mylist = [['pielou', {'bootstrap': True, 'ccp_alpha': 0.0, 'class_weight': None, 'criterion': 'gini'}, {'accuracy':77, 'precision':80}], ['genus', {'bootstrap': True, 'ccp_alpha': 0.5, 'class_weight': 2, 'criterion': 'gini'}, {'accuracy':74, 'precision':69}]]
    results_table = pd.DataFrame()
    df = pd.DataFrame(columns=['Normalization'])
    for i in mylist:
        # Create a dataframe with 'Normalization' as the first column
        df.loc[0] = i[0]
        # Iterate through the rest of the list and add the key-value pairs as columns and rows
        for d in i[1:]:
            for key, value in d.items():
                if key not in df.columns:
                    df[key] = None
                df.loc[0, key] = value
        results_table = results_table.append(df)


    full_results = []
    data = load_tsv_files(filepath)
    for key in data:
        X_train, X_test, y_train, y_test = preprocess_data(data[key], yang_metadata_path) #preprocess_fudan_data?
        scores, best_results, best_estimator = grid_search_rf(X_train, X_test, y_train, y_test, file_name=key)
        full_results.append([str(key),best_estimator, best_results])
        visualize_results(scores, data_name, key)
        print(full_results)

    results_table = pd.DataFrame()
    df = pd.DataFrame(columns=['Normalization'])
    for i in full_results:
        # Create a dataframe with 'Normalization' as the first column
        df.loc[0] = i[0]
        # Iterate through the rest of the list and add the key-value pairs as columns and rows
        for d in i[1:]:
            for key, value in d.items():
                if key not in df.columns:
                    df[key] = None
                df.loc[0, key] = value
        results_table = results_table.append(df)
        results_table.drop(['oob_score', 'min_weight_fraction_leaf', 'bootstrap', 'ccp_alpha'], inplace=True, axis=1)



    # split 'metrics' column into separate columns
    #df_metrics = pd.json_normalize(df['metrics'])

    # join the new DataFrame with the original one
    #df = df.join(df_metrics)

    # drop the 'metrics' column
    #df.drop(['metrics'], axis=1, inplace=True)

    latex_table = results_table.to_latex()

    with open(os.path.join(Config.LOG_DIR, data_name, f"best_results.txt"), "w") as f:
        f.write(latex_table)
    print("End of file 1")


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=FUDAN)  # attack type
    parser.add_argument('--filepath', type=str, default=fudan_filepath)     # number of LSB set to secrets
    #parser.add_argument('--n', type=int, default=100)      # number of data points to be encoded in LSB
    #parser.add_argument('--cr', type=float, default=1.0)    # malicious term ratio
    #parser.add_argument('--p', type=float, default=1.0)     # proportion of malicious data to training data
    #parser.add_argument('--model', type=int, default=1)     # number of blocks in resnet
    #parser.add_argument('--lsb_def_bits', type=int, default=1)  # number of replaced bits by defense
    #parser.add_argument('--gzip_incl', type=bool, default=False) #including gzipping in image compression/decompression

    args = parser.parse_args()
    data_name = args.data_name
    if data_name == FUDAN:
        run_rf_tuning(data_name=args.data_name, filepath=args.filepath)
    elif data_name == HUADONG1:
        run_rf_tuning(data_name=args.data_name, filepath=args.filepath)
    #    test_cor_reconstruction(cr=args.cr, res_n=args.model)
    #elif attack == SGN:
    #    test_sgn_reconstruction(cr=args.cr, res_n=args.model)
    #elif attack == LSB:
        #test_lsb_acc(bits=args.bits, n_data=args.n, res_n=args.model, gzip_incl=args.gzip_incl)
        #test_lsb_reconstruction(bits=args.bits, n_data=args.n, res_n=args.model, gzip_incl=args.gzip_incl)
        #test_lsb_defense_acc(bits=args.bits, lsb_def_bits=args.lsb_def_bits, n_data=args.n, res_n=args.model, gzip_incl=args.gzip_incl)
        #test_lsb_defense_effectiveness(bits=args.bits, lsb_def_bits = args.lsb_def_bits, n_data=args.n, res_n=args.model, gzip_incl=args.gzip_incl)
    else:
        raise ValueError()


