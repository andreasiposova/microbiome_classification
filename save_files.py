import os.path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, roc_auc_score, accuracy_score
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.preprocessing import MinMaxScaler

from data_loading import load_young_old_labels, load_tsv_files
from feature_selection import select_features_from_paper
from preprocessing import full_preprocessing_y_o_labels, preprocess_data, preprocess_huadong
from utils import Config

FUDAN = 'fudan'
HUADONG1 = 'huadong1'
HUADONG2 = 'huadong2'

file_names = list(("pielou_e_diversity", "simpson_diversity", "phylum_relative", "observed_otus_diversity", "family_relative", "class_relative", "fb_ratio", "enterotype", "genus_relative", "species_relative", "shannon_diversity", "domain_relative", "order_relative", "simpson_e_diversity"))

yang_metadata_path = "data/Yang_PRJNA763023/metadata.csv"
fudan_filepath = 'data/Yang_PRJNA763023/Yang_PRJNA763023_SE/parsed/normalized_results/'
huadong_filepath_1 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_1/parsed/normalized_results'
huadong_filepath_2 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_2/parsed/normalized_results'
young_old_labels_path = 'data/Yang_PRJNA763023/SraRunTable.csv'




def load_preprocessed_data(data_name, filepath, group, select_features):
    y_o_labels = load_young_old_labels(young_old_labels_path)
    data = load_tsv_files(filepath)
    huadong_data1 = load_tsv_files(huadong_filepath_1)
    huadong_data2 = load_tsv_files(huadong_filepath_2)

    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    X_val = pd.DataFrame()

    for key in data:
        # if key == "genus_relative" or key == "family_relative":
        # X_train, X_test, y_train, y_test = preprocess_data(data[key], yang_metadata_path) #preprocess_fudan_data?
        if group == "all":
            X_train_1, X_test_1, y_train, y_test = preprocess_data(data[key], yang_metadata_path)
            X_h1, y_h1 = preprocess_huadong(huadong_data1[key], yang_metadata_path)
            X_h2, y_h2 = preprocess_huadong(huadong_data2[key], yang_metadata_path)
            X_val_1 = pd.concat([X_h1, X_h2])
            y_val = y_h1 + y_h2
        elif group == 'young' or group == 'old':
            X_train_1, X_test_1, X_val_1, y_train, y_test, y_val = full_preprocessing_y_o_labels(data, huadong_data1,
                                                                                                 huadong_data2, key,
                                                                                                 yang_metadata_path,
                                                                                                 young_old_labels_path,
                                                                                                 group)

        X_train = pd.concat([X_train, X_train_1], axis=1)
        #X_test = pd.concat([X_test, X_test_1], axis=1)
        X_val = pd.concat([X_val, X_val_1], axis=1)

    if select_features == False:
        file_name = "all_features"

    if select_features == True:
        file_name = "selected_features"
        # top_features = calculate_feature_importance(X_train, y_train, group)
        # top_features_names = list(map(lambda x: x[0], top_features))
        # print(top_features_names)
        # X_train = X_train[top_features_names]
        # X_train.to_csv('data/selected_features_old.csv')
        # common_cols_f = set(X_test.columns).intersection(X_train.columns)
        # common_cols_fv = set(X_val.columns).intersection(X_train.columns)
        # X_test = X_test[common_cols_f]
        # X_val = X_val[common_cols_fv]
        #X_test = select_features_from_paper(X_test, group, key)
        X_train = select_features_from_paper(X_train, group, key)
        X_val = select_features_from_paper(X_val, group, key)

    #common_cols_t = set(X_test.columns).intersection(X_val.columns)
    common_cols_v = set(X_val.columns).intersection(X_train.columns)

    # filling missing values in huadong cohort with zeros
    # as two files are concatenated for huadong cohort files
    # they contain columns that are not compatible
    # thus creating missing values - they are replaced with 0 as it means the abundace of that bacteria is anyway 0
    X_val = X_val.fillna(0)
    X_val = X_val[common_cols_v]
    X_train = X_train[common_cols_v]
    #X_test = X_test[common_cols_t]
    #X_train = X_train.append(X_test)
    #y_train = y_train + y_test
    # corr = X_train.corr()
    # X_train = remove_correlated_features(X_train, 0.95)
    # common_cols_t = set(X_test.columns).intersection(X_train.columns)
    # common_cols_v = set(X_val.columns).intersection(X_train.columns)
    # X_val = X_val[common_cols_v]
    # X_test = X_test[common_cols_t]

    print("number of features: ", X_train.shape[1])
    print("number of samples in training set: ", len(X_train))
    print("number of samples in test set: ", len(X_test))
    print("number of samples in validation set: ", len(X_val))

    print(f"Running experiments on {group} samples")

    #scaler = MinMaxScaler()
    #scaler.fit(X_train)
    #X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    #X_val = scaler.transform(X_val)

    return X_train, X_test, X_val, y_train, y_test, y_val


def save_data(FUDAN, fudan_filepath, group, select_features):

    if select_features == True:
        if not os.path.exists(str(Config.DATA_DIR) + f"/preprocessed/{group}/selected_features"):
            os.makedirs(os.path.join(Config.DATA_DIR, 'preprocessed', group, "selected_features"))
        X_train, X_test, X_val, y_train, y_test, y_val = load_preprocessed_data(FUDAN, fudan_filepath, group, select_features)
        y_train=pd.DataFrame(y_train)
        y_val = pd.DataFrame(y_val)
        X_train.to_csv(os.path.join(Config.DATA_DIR, 'preprocessed',  group, f"selected_features/X_train.csv"))
        X_val.to_csv(os.path.join(Config.DATA_DIR, 'preprocessed',  group, f"selected_features/X_val.csv"))
        y_train.to_csv(os.path.join(Config.DATA_DIR, 'preprocessed',  group, f"selected_features/y_train.csv"))
        y_val.to_csv(os.path.join(Config.DATA_DIR, 'preprocessed',  group, f"selected_features/y_val.csv"))
    if select_features == False:
        if not os.path.exists(str(Config.DATA_DIR) + f"/preprocessed/{group}/all_features"):
            os.makedirs(os.path.join(Config.DATA_DIR, 'preprocessed', group, "all_features"))
        X_train, X_test, X_val, y_train, y_test, y_val = load_preprocessed_data(FUDAN, fudan_filepath, group, select_features)
        y_train = pd.DataFrame(y_train)
        y_val = pd.DataFrame(y_val)
        X_train.to_csv(os.path.join(Config.DATA_DIR, 'preprocessed', group, f"all_features/X_train.csv"))
        X_val.to_csv(os.path.join(Config.DATA_DIR, 'preprocessed', group, f"all_features/X_val.csv"))
        y_train.to_csv(os.path.join(Config.DATA_DIR, 'preprocessed', group, f"all_features/y_train.csv"))
        y_val.to_csv(os.path.join(Config.DATA_DIR, 'preprocessed', group, f"all_features/y_val.csv"))

#save_data(FUDAN, fudan_filepath, group= 'young', select_features=True)
#save_data(FUDAN, fudan_filepath, group= 'young', select_features=False)
#save_data(FUDAN, fudan_filepath, group= 'old', select_features=True)
#save_data(FUDAN, fudan_filepath, group= 'old', select_features=False)
#save_data(FUDAN, fudan_filepath, group= 'all', select_features=True)
#save_data(FUDAN, fudan_filepath, group= 'all', select_features=False)

def load_Xy(group, select_features):
    if select_features == True:
        X_train = pd.read_csv(os.path.join(Config.DATA_DIR, 'preprocessed', group, 'selected_features/X_train.csv'), index_col=0)
        y_train = pd.read_csv(os.path.join(Config.DATA_DIR, 'preprocessed', group, 'selected_features/y_train.csv'),index_col=0)
        X_val = pd.read_csv(os.path.join(Config.DATA_DIR,'preprocessed', group, 'selected_features/X_val.csv'), index_col=0)
        y_val = pd.read_csv(os.path.join(Config.DATA_DIR, 'preprocessed', group, 'selected_features/y_val.csv'), index_col = 0)
    if select_features == False:
        X_train = pd.read_csv(os.path.join(Config.DATA_DIR, 'preprocessed', group, 'all_features/X_train.csv'), index_col=0)
        y_train = pd.read_csv(os.path.join(Config.DATA_DIR, 'preprocessed', group, 'all_features/y_train.csv'), index_col=0)
        X_val = pd.read_csv(os.path.join(Config.DATA_DIR,'preprocessed', group, 'all_features/X_val.csv'), index_col = 0)
        y_val = pd.read_csv(os.path.join(Config.DATA_DIR, 'preprocessed', group, 'all_features/y_val.csv'), index_col=0)
    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_val = scaler.transform(X_val)
    y_train = y_train.iloc[:,0].to_numpy()
    y_val = y_val.iloc[:,0].to_numpy()

    param_grid = {
        'n_estimators': np.arange(10, 40, 5, dtype=int),
        'max_depth': np.arange(5, 10, 1, dtype=int),
        'min_samples_split': np.arange(7, 20, 2, dtype=int),
        'min_samples_leaf': np.arange(4, 12, 1, dtype=int),
        'max_features': ['sqrt', 'log2'],  # 'sqrt', 'log2'],
        'random_state': [1234],
        'class_weight': ['balanced_subsample']
    }

    scoring = {
        'roc_auc': make_scorer(roc_auc_score),
        'accuracy': make_scorer(accuracy_score),
        #'precision': make_scorer(accuracy_score),
        #'f1': make_scorer(f1_score)
        #'recall': make_scorer(recall_score)
    }

    rf = RandomForestClassifier(random_state=1234)
    train_scores_gridsearch = []
    test_scores_gridsearch = []
    cv = KFold(n_splits=2, random_state=1234, shuffle=True)
    # Perform the grid search
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=cv, scoring=scoring, refit='roc_auc',
                               return_train_score=True, n_jobs=-1)
    grid_search.fit(X_train, y_train)
    results = grid_search.cv_results_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_
    grid_val_scores=pd.DataFrame()
    for i in range(len(grid_search.cv_results_['params'])):
        rf_ = RandomForestClassifier(**grid_search.cv_results_['params'][i])
        rf_.fit(X_train, y_train)
        y_pred = rf_.predict(X_val)
        accuracy = accuracy_score(y_val, y_pred)
        #precision = precision_score(y_val, y_pred)
        #recall = recall_score(y_val, y_pred)
        roc_auc = roc_auc_score(y_val, y_pred)
        if roc_auc > 0.68:
            print(roc_auc)
            print(grid_search.cv_results_['params'][i])
    """print("hi")
        grid_val_scores = grid_val_scores.append({'params': grid_search.cv_results_['params'][i],
                                                   'accuracy': accuracy,
                                #                   'precision': precision,
                                #                  'recall': recall,
                                                  'roc_auc': roc_auc}, ignore_index=True)
                                #                  'f1': f1, 'f2': f2})
    results_auroc = [d for d in grid_val_scores if 0.1 <= d['roc_auc'] <= 1.0]
    best_auroc_params = max(results_auroc, key= lambda x: x['roc_auc'])
    best_params = best_auroc_params['params']"""
    #print(best_params)
    #print(best_score)

load_Xy('young', False)