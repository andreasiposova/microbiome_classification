import json
import os
import ast
import numpy as np
from math import sqrt

import forestci as fci
import pandas as pd
import xgboost
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, fbeta_score, \
    confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import MinMaxScaler

from data_loading import load_tsv_files, load_young_old_labels
from feature_selection import select_features_from_paper
from preprocessing import preprocess_data, preprocess_huadong, full_preprocessing_y_o_labels, \
    apply_feature_abundance_limits
from utils import Config

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt
np.random.seed(43)

from visualization import plot_conf_int, cm_plot, prob_boxplot

FUDAN = 'fudan'
HUADONG1 = 'huadong1'
HUADONG2 = 'huadong2'

file_names = list(("pielou_e_diversity", "simpson_diversity", "phylum_relative", "observed_otus_diversity", "family_relative", "class_relative", "fb_ratio", "enterotype", "genus_relative", "species_relative", "shannon_diversity", "domain_relative", "order_relative", "simpson_e_diversity"))

yang_metadata_path = "data/Yang_PRJNA763023/metadata.csv"
fudan_filepath = 'data/Yang_PRJNA763023/Yang_PRJNA763023_SE/parsed/normalized_results/'
huadong_filepath_1 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_1/parsed/normalized_results'
huadong_filepath_2 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_2/parsed/normalized_results'
young_old_labels_path = 'data/Yang_PRJNA763023/SraRunTable.csv'



def get_best_params(data_name, group, file_name, clf_name, param_file = 'best_params'):
    with open(os.path.join(Config.LOG_DIR, data_name, group, file_name, f"{clf_name}_{param_file}.txt"), "rb") as f:
        params = f.read()
        params = ast.literal_eval(params.decode())
        params = params#['params']
    return params



def load_preprocessed_data(data_name=FUDAN, filepath=fudan_filepath, group='old', select_features=True):
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

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    #X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    return X_train, X_test, X_val, y_train, y_test, y_val


def perform_rf_classification(X_train, X_test, X_val, y_train, y_test, y_val, params, group, file_name, fal, fal_type):

    results_df = pd.DataFrame()


    clf = RandomForestClassifier(**params)
    #clf.fit(X_train, y_train)
    #y_train_pred = clf.predict(X_train)
    #y_test_pred = clf.predict(X_test)
    #y_val_pred = clf.predict(X_val)
    threshold = 0.50
    predictions_crc = pd.DataFrame()
    predictions_h = pd.DataFrame()
    n_runs = 10
    train_results = pd.DataFrame()
    # test_results = pd.DataFrame()
    val_results = pd.DataFrame()
    #for i in range(n_runs):

    clf.fit(X_train, y_train)
    pred = clf.predict_proba(X_train)
    #prob_crc = pred[:,-1]
    #df_crc = pd.DataFrame(prob_crc)
    #predictions_crc = predictions_crc.append(df_crc, ignore_index=False)
    #predictions_crc = pd.concat([predictions_crc, df_crc], axis=1)
    #prob_h = pred[:,0:1]
    #df_h = pd.DataFrame(prob_h)
    #predictions_h = pd.concat([predictions_h, df_h], axis=1)#predictions_h = predictions_h.append(df_h, ignore_index=False)


    y_train_prob = clf.predict_proba(X_train)
    prob_boxplot(y_train, y_train_prob, FUDAN, group, file_name, 'train', fal, fal_type)
    y_train_pred = (y_train_prob[:, 1] >= threshold).astype('int')
    plot_conf_int(y_train, y_train_prob, X_train, X_train, clf, data_name=FUDAN, file_name=file_name, group=group, set_name="train", fal=fal, fal_type=fal_type)


    #y_test_prob = clf.predict_proba(X_test)
    #y_test_pred = (y_test_prob[:, 1] >= threshold).astype('int')
    #plot_conf_int(y_test, y_test_prob, X_train, X_test, clf, data_name=FUDAN, file_name=file_name, group=group, set_name="test")

    y_val_prob = clf.predict_proba(X_val)
    y_val_pred = (y_val_prob[:, 1] >= threshold).astype('int')
    prob_boxplot(y_val, y_val_prob, FUDAN, group, file_name, 'val', fal, fal_type)
    plot_conf_int(y_val, y_val_prob, X_train, X_val, clf, data_name=FUDAN, file_name=file_name, group=group, set_name="val", fal=fal, fal_type=fal_type)

    acc_train = accuracy_score(y_train, y_train_pred)
    prec_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    roc_auc_train = roc_auc_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred)
    f2_train = fbeta_score(y_train, y_train_pred, beta=2)
    interval_len = 1.96 * sqrt((roc_auc_train * (1 - roc_auc_train)) / len(y_train))
    interval_low = roc_auc_train - interval_len
    interval_high = roc_auc_train + interval_len
    ci_train = [interval_low, interval_high]
    cm = confusion_matrix(y_train, y_train_pred)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    train_set_res = {'roc_auc': [roc_auc_train], 'CI lower': [interval_low], 'CI upper': [interval_high],
                     'accuracy': [acc_train], 'precision': [prec_train], 'recall': [recall_train], 'f1': [f1_train],
                     'f2': [f2_train], 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
    train_set_res = pd.DataFrame.from_dict(train_set_res)
    train_results = pd.concat([val_results, train_set_res], axis=1)


    #acc_test = accuracy_score(y_test, y_test_pred)
    #prec_test = precision_score(y_test, y_test_pred)
    #recall_test = recall_score(y_test, y_test_pred)
    #roc_auc_test = roc_auc_score(y_test, y_test_pred)
    #f1_test = f1_score(y_test, y_test_pred)
    #f2_test = fbeta_score(y_test, y_test_pred, beta=2)
    #interval_len = 1.96 * sqrt((roc_auc_test * (1 - roc_auc_test)) / len(y_test))
    #interval_low = roc_auc_test - interval_len
    #interval_high = roc_auc_test + interval_len
    #ci_test = [interval_low, interval_high]
    #test_set_res = {'accuracy': acc_test, 'precision': prec_test, 'recall': recall_test,
     #                     'roc_auc': roc_auc_test, 'f1': f1_test,
     #                     'f2': f2_test, 'auc_conf_int': ci_test}
    #test_results = test_results.append(test_set_res, ignore_index=True)



    acc_val = accuracy_score(y_val, y_val_pred)
    prec_val = precision_score(y_val, y_val_pred)
    recall_val = recall_score(y_val, y_val_pred)
    roc_auc_val = roc_auc_score(y_val, y_val_pred)
    f1_val = f1_score(y_val, y_val_pred)
    f2_val = fbeta_score(y_val, y_val_pred, beta=2)
    interval_len = 1.96 * sqrt((roc_auc_val * (1 - roc_auc_val)) / len(y_val))
    interval_low = roc_auc_val - interval_len
    interval_high = roc_auc_val + interval_len
    ci_val = [interval_low, interval_high]
    cm = confusion_matrix(y_val, y_val_pred)
    cm_plot(y_val, y_val_pred, 'fudan', group, file_name, 'val', 'RF', final=True, fal=fal, fal_type=fal_type)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    val_set_res = {'roc_auc': [roc_auc_val], 'CI lower': [interval_low], 'CI upper': [interval_high],
                   'accuracy': [acc_val], 'precision': [prec_val], 'recall': [recall_val],
                   'f1': [f1_val], 'f2': [f2_val], 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
    val_set_res = pd.DataFrame.from_dict(val_set_res)
    val_results = pd.concat([val_results, val_set_res], axis=1)

    train_results = train_results.mean()
    #test_results = test_results.mean()
    val_results = val_results.mean()

    results_df = results_df.append(train_results, ignore_index=True)
    #results_df = results_df.append(test_results, ignore_index=True)
    results_df = results_df.append(val_results, ignore_index=True)

    return results_df

def perform_classification(X_train, X_test, X_val, y_train, y_test, y_val, params, group, file_name, clf_name, fal, fal_type):

    results_df = pd.DataFrame()
    if clf_name == "SVM":
        clf = svm.SVC(**params)
    elif clf_name == "XGB":
        clf = xgboost.XGBClassifier(**params)
    elif clf_name == "KNN":
        clf = KNeighborsClassifier(**params)

    clf.fit(X_train, y_train)

    train_results = pd.DataFrame()
    # test_results = pd.DataFrame()
    val_results = pd.DataFrame()
    clf.fit(X_train, y_train)


    y_train_pred = clf.predict(X_train)

    y_val_pred = clf.predict(X_val)

    acc_train = accuracy_score(y_train, y_train_pred)
    prec_train = precision_score(y_train, y_train_pred)
    recall_train = recall_score(y_train, y_train_pred)
    roc_auc_train = roc_auc_score(y_train, y_train_pred)
    f1_train = f1_score(y_train, y_train_pred)
    f2_train = fbeta_score(y_train, y_train_pred, beta=2)
    interval_len = 1.96 * sqrt((roc_auc_train * (1 - roc_auc_train)) / len(y_train))
    interval_low = roc_auc_train - interval_len
    interval_high = roc_auc_train + interval_len
    ci_train = [interval_low, interval_high]
    cm = confusion_matrix(y_train, y_train_pred)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    train_set_res = {'roc_auc': [roc_auc_train], 'CI lower': [interval_low], 'CI upper': [interval_high], 'accuracy': [acc_train], 'precision': [prec_train], 'recall': [recall_train], 'f1': [f1_train],
                          'f2': [f2_train], 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
    train_set_res = pd.DataFrame.from_dict(train_set_res)
    train_results = pd.concat([val_results, train_set_res], axis=1)
    #train_results = train_results.append(train_set_res, ignore_index=True)


    #acc_test = accuracy_score(y_test, y_test_pred)
    #prec_test = precision_score(y_test, y_test_pred)
    #recall_test = recall_score(y_test, y_test_pred)
    #roc_auc_test = roc_auc_score(y_test, y_test_pred)
    #f1_test = f1_score(y_test, y_test_pred)
    #f2_test = fbeta_score(y_test, y_test_pred, beta=2)
    #interval_len = 1.96 * sqrt((roc_auc_test * (1 - roc_auc_test)) / len(y_test))
    #interval_low = roc_auc_test - interval_len
    #interval_high = roc_auc_test + interval_len
    #ci_test = [interval_low, interval_high]
    #test_set_res = {'accuracy': acc_test, 'precision': prec_test, 'recall': recall_test,
     #                     'roc_auc': roc_auc_test, 'f1': f1_test,
     #                     'f2': f2_test, 'auc_conf_int': ci_test}
    #test_results = test_results.append(test_set_res, ignore_index=True)


    acc_val = accuracy_score(y_val, y_val_pred)
    prec_val = precision_score(y_val, y_val_pred)
    recall_val = recall_score(y_val, y_val_pred)
    roc_auc_val = roc_auc_score(y_val, y_val_pred)
    f1_val = f1_score(y_val, y_val_pred)
    f2_val = fbeta_score(y_val, y_val_pred, beta=2)
    interval_len = 1.96 * sqrt((roc_auc_val * (1 - roc_auc_val)) / len(y_val))
    interval_low = roc_auc_val - interval_len
    interval_high = roc_auc_val + interval_len
    ci_val = [interval_low, interval_high]
    cm = confusion_matrix(y_val, y_val_pred)
    print(cm)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
    cm_plot(y_val, y_val_pred, 'fudan', group, file_name, 'val', clf_name, final=True, fal=fal, fal_type=fal_type)
    val_set_res = {'roc_auc': [roc_auc_val], 'CI lower': [interval_low], 'CI upper': [interval_high], 'accuracy': [acc_val], 'precision': [prec_val], 'recall': [recall_val],
                    'f1': [f1_val], 'f2': [f2_val], 'TP': TP, 'TN': TN, 'FP': FP, 'FN': FN}
    val_set_res = pd.DataFrame.from_dict(val_set_res)
    val_results = pd.concat([val_results, val_set_res], axis=1)
    #val_results = val_results.append(val_set_res, ignore_index=True)

    train_results = train_results #.mean()
    #test_results = test_results.mean()
    val_results = val_results #.mean()

    results_df = results_df.append(train_results, ignore_index=True)
    #results_df = results_df.append(test_results, ignore_index=True)
    results_df = results_df.append(val_results, ignore_index=True)


    return results_df




def get_results(data_name, filepath, group, select_features, clf_name, fal, fal_type):
    X_train, X_test, X_val, y_train, y_test, y_val = load_preprocessed_data(FUDAN, fudan_filepath, group, select_features)
    if select_features == True:
        params = get_best_params(FUDAN, group, 'selected_features', clf_name, param_file='best_params')
        if fal == True:
            X_train = apply_feature_abundance_limits(X_train, fal_type)
            X_test = apply_feature_abundance_limits(X_test, fal_type)
            X_val = apply_feature_abundance_limits(X_val, fal_type)
            if clf_name== "RF":
                results = perform_rf_classification(X_train, X_test, X_val, y_train, y_test, y_val, params, group=group, file_name="selected_features", fal=True, fal_type = fal_type)
                settings = {'classifier': [clf_name], 'samples': [group], 'features': ['selected'], 'Feature Abundance Limits': [fal_type]}
            else:
                results = perform_classification(X_train, X_test, X_val, y_train, y_test, y_val, params, group=group, file_name="selected_features", clf_name=clf_name, fal=True, fal_type = fal_type)
                settings = {'classifier': [clf_name], 'samples': [group], 'features': ['selected'],
                            'Feature Abundance Limits': [fal_type]}

        if fal == False:
            if clf_name == "RF":
                results = perform_rf_classification(X_train, X_test, X_val, y_train, y_test, y_val, params, group=group, file_name="selected_features", fal = False, fal_type=None)
                settings = {'classifier': [clf_name], 'samples': [group], 'features': ['selected'],
                            'Feature Abundance Limits': ['no transformation']}
            else:
                results = perform_classification(X_train, X_test, X_val, y_train, y_test, y_val, params, group=group, file_name="selected_features", clf_name=clf_name, fal=False, fal_type = None)
                settings = {'classifier': [clf_name], 'samples': [group], 'features': ['selected'],
                            'Feature Abundance Limits': ['no transformation']}

    if select_features == False:
        params = get_best_params(FUDAN, group, 'all_features', clf_name,  param_file='best_params')
        if fal == True:
            X_train = apply_feature_abundance_limits(X_train, fal_type)
            X_test = apply_feature_abundance_limits(X_test, fal_type)
            X_val = apply_feature_abundance_limits(X_val, fal_type)
            if clf_name == "RF":
                results = perform_rf_classification(X_train, X_test, X_val, y_train, y_test, y_val, params,
                                                    group=group, file_name="all_features", fal=True, fal_type=fal_type)
                settings = {'classifier': [clf_name], 'samples': [group], 'features': ['all'],
                            'Feature Abundance Limits': [fal_type]}
            else:
                results = perform_classification(X_train, X_test, X_val, y_train, y_test, y_val, params,
                                                 group=group, file_name="all_features", clf_name=clf_name, fal=True, fal_type = fal_type)
                settings = {'classifier': [clf_name], 'samples': [group], 'features': ['all'],
                            'Feature Abundance Limits': [fal_type]}

        if fal == False:
            if clf_name == "RF":
                results = perform_rf_classification(X_train, X_test, X_val, y_train, y_test, y_val, params,
                                                    group=group, file_name="all_features", fal=False, fal_type=None)
                settings = {'classifier': [clf_name], 'samples': [group], 'features': ['all'],
                            'Feature Abundance Limits': ['no transformation']}
            else:
                results = perform_classification(X_train, X_test, X_val, y_train, y_test, y_val, params,
                                                 group=group, file_name="all_features", clf_name=clf_name, fal=False, fal_type = None)
                settings = {'classifier': [clf_name], 'samples': [group], 'features': ['all'],
                            'Feature Abundance Limits': ['no transformation']}


    settings = pd.DataFrame.from_dict(settings)
    results = results.iloc[[1]]
    results = results.reset_index(drop=True)
    results = pd.concat([settings, results], axis=1)
    print(results)
    return results

def svm_results(data_name=FUDAN, fudan_filepath = fudan_filepath):
    #OLD SVM ___ SELECTED FEATURES ___
    otf = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name = "SVM", fal=False, fal_type='high')
    otl = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name = "SVM", fal=True, fal_type='low')
    otm = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name = "SVM", fal=True, fal_type='medium')
    oth = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name = "SVM", fal=True, fal_type='high')

    #OLD SVM ___ ALL FEATURES ___
    off = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name = "SVM", fal=False, fal_type='high')
    ofl = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name = "SVM", fal=True, fal_type='low')
    ofm = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name = "SVM", fal=True, fal_type='medium')
    ofh = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name = "SVM", fal=True, fal_type='high')

    #YOUNG SVM ___ SELECTED FEATURES ___
    ytf = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name = "SVM", fal=False, fal_type='high')
    ytl = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name = "SVM", fal=True, fal_type='low')
    ytm = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name = "SVM", fal=True, fal_type='medium')
    yth = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name = "SVM", fal=True, fal_type='high')

    #YOUNG SVM ___ ALL FEATURES ___
    yff = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name = "SVM", fal=False, fal_type='high')
    yfl = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name = "SVM", fal=True, fal_type='low')
    yfm = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name = "SVM", fal=True, fal_type='medium')
    yfh = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name = "SVM", fal=True, fal_type='high')

    #ALL SVM ___ SELECTED FEATURES ___
    atf = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name = "SVM", fal=False, fal_type='high')
    atl = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name = "SVM", fal=True, fal_type='low')
    atm = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name = "SVM", fal=True, fal_type='medium')
    ath = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name = "SVM", fal=True, fal_type='high')

    #ALL SVM ___ ALL FEATURES ___
    aff = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name = "SVM", fal=False, fal_type='high')
    afl = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name = "SVM", fal=True, fal_type='low')
    afm = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name = "SVM", fal=True, fal_type='medium')
    afh = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name = "SVM", fal=True, fal_type='high')

    svm_res = pd.concat([otf, otl, otm, oth, off, ofl, ofm, ofh, ytf, ytl, ytm, yth, yff, yfl, yfm, yfh, atf, atl, atm, ath, aff, afl, afm, afh])
    if not os.path.exists(str(Config.LOG_DIR) + "/" + FUDAN + "/final_results/"):
        os.makedirs(os.path.join(Config.LOG_DIR, FUDAN, 'final_results'))
    svm_res.to_csv(os.path.join(Config.LOG_DIR, FUDAN, 'final_results/svm_final_results.csv'))
    return svm_res


def rf_results(data_name=FUDAN, fudan_filepath=fudan_filepath):
    # OLD RF ___ SELECTED FEATURES ___
    #otf = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name="RF", fal=False,fal_type='high')
    #otl = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name="RF", fal=True, fal_type='low')
    #otm = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name="RF", fal=True,fal_type='medium')
    #oth = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name="RF", fal=True, fal_type='high')

    # OLD RF ___ ALL FEATURES ___
    #off = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name="RF", fal=False,fal_type='high')
    #ofl = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name="RF", fal=True, fal_type='low')
    #ofm = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name="RF", fal=True,fal_type='medium')
    #ofh = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name="RF", fal=True,fal_type='high')

    # YOUNG RF ___ SELECTED FEATURES ___
    #ytf = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name="RF", fal=False,fal_type='high')
    #ytl = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name="RF", fal=True,fal_type='low')
    #ytm = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name="RF", fal=True,fal_type='medium')
    #yth = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name="RF", fal=True,fal_type='high')

    # YOUNG RF ___ ALL FEATURES ___
    #yff = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name="RF", fal=False, fal_type='high')
    #yfl = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name="RF", fal=True, fal_type='low')
    #yfm = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name="RF", fal=True,fal_type='medium')
    #yfh = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name="RF", fal=True,fal_type='high')

    # ALL RF ___ SELECTED FEATURES ___
    #atf = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name="RF", fal=False,fal_type='high')
    #atl = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name="RF", fal=True, fal_type='low')
    #atm = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name="RF", fal=True,fal_type='medium')
    #ath = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name="RF", fal=True, fal_type='high')

    # ALL RF ___ ALL FEATURES ___
    aff = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name="RF", fal=False,fal_type='high')
    afl = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name="RF", fal=True, fal_type='low')
    afm = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name="RF", fal=True,fal_type='medium')
    afh = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name="RF", fal=True,fal_type='high')
    rf_res = pd.concat([otf, otl, otm, oth, off, ofl, ofm, ofh, ytf, ytl, ytm, yth, yff, yfl, yfm, yfh, atf, atl, atm, ath, aff, afl, afm, afh])
    if not os.path.exists(str(Config.LOG_DIR) + "/" + FUDAN + "/final_results/"):
        os.makedirs(os.path.join(Config.LOG_DIR, FUDAN, 'final_results'))
    rf_res.to_csv(os.path.join(Config.LOG_DIR, FUDAN, 'final_results/rf_final_results.csv'))

    return rf_res


def knn_results(data_name=FUDAN, fudan_filepath=fudan_filepath):
    # OLD KNN ___ SELECTED FEATURES ___
    otf = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name="KNN", fal=False,fal_type='high')
    otl = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name="KNN", fal=True, fal_type='low')
    otm = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name="KNN", fal=True,fal_type='medium')
    oth = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name="KNN", fal=True, fal_type='high')

    # OLD KNN ___ ALL FEATURES ___
    off = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name="KNN", fal=False,fal_type='high')
    ofl = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name="KNN", fal=True, fal_type='low')
    ofm = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name="KNN", fal=True,fal_type='medium')
    ofh = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name="KNN", fal=True,fal_type='high')

    # YOUNG KNN ___ SELECTED FEATURES ___
    ytf = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name="KNN", fal=False,fal_type='high')
    ytl = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name="KNN", fal=True,fal_type='low')
    ytm = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name="KNN", fal=True,fal_type='medium')
    yth = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name="KNN", fal=True,fal_type='high')

    # YOUNG KNN ___ ALL FEATURES ___
    yff = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name="KNN", fal=False,fal_type='high')
    yfl = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name="KNN", fal=True,fal_type='low')
    yfm = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name="KNN", fal=True,fal_type='medium')
    yfh = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name="KNN", fal=True,fal_type='high')

    # ALL KNN ___ SELECTED FEATURES ___
    atf = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name="KNN", fal=False,fal_type='high')
    atl = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name="KNN", fal=True, fal_type='low')
    atm = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name="KNN", fal=True,fal_type='medium')
    ath = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name="KNN", fal=True, fal_type='high')

    # ALL KNN ___ ALL FEATURES ___
    aff = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name="KNN", fal=False,fal_type='high')
    afl = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name="KNN", fal=True, fal_type='low')
    afm = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name="KNN", fal=True,fal_type='medium')
    afh = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name="KNN", fal=True,fal_type='high')
    knn_res = pd.concat([otf, otl, otm, oth, off, ofl, ofm, ofh, ytf, ytl, ytm, yth, yff, yfl, yfm, yfh, atf, atl, atm, ath, aff, afl, afm, afh])
    if not os.path.exists(str(Config.LOG_DIR) + "/" + FUDAN + "/final_results/"):
        os.makedirs(os.path.join(Config.LOG_DIR, FUDAN, 'final_results'))
    knn_res.to_csv(os.path.join(Config.LOG_DIR, FUDAN, 'final_results/knn_final_results.csv'))
    return knn_res


def xgb_results(data_name=FUDAN, fudan_filepath=fudan_filepath):
    # OLD XGB ___ SELECTED FEATURES ___
    otf = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name="XGB", fal=False,fal_type='high')
    otl = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name="XGB", fal=True, fal_type='low')
    otm = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name="XGB", fal=True,fal_type='medium')
    oth = get_results(data_name, fudan_filepath, 'old', select_features=True, clf_name="XGB", fal=True, fal_type='high')

    # OLD XGB ___ ALL FEATURES ___
    off = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name="XGB", fal=False,fal_type='high')
    ofl = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name="XGB", fal=True, fal_type='low')
    ofm = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name="XGB", fal=True,fal_type='medium')
    ofh = get_results(data_name, fudan_filepath, 'old', select_features=False, clf_name="XGB", fal=True,fal_type='high')

    # YOUNG XGB ___ SELECTED FEATURES ___
    ytf = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name="XGB", fal=False,fal_type='high')
    ytl = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name="XGB", fal=True,fal_type='low')
    ytm = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name="XGB", fal=True,fal_type='medium')
    yth = get_results(data_name, fudan_filepath, 'young', select_features=True, clf_name="XGB", fal=True,fal_type='high')

    # YOUNG XGB ___ ALL FEATURES ___
    yff = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name="XGB", fal=False,fal_type='high')
    yfl = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name="XGB", fal=True,fal_type='low')
    yfm = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name="XGB", fal=True,fal_type='medium')
    yfh = get_results(data_name, fudan_filepath, 'young', select_features=False, clf_name="XGB", fal=True,fal_type='high')

    # ALL XGB ___ SELECTED FEATURES ___
    atf = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name="XGB", fal=False,fal_type='high')
    atl = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name="XGB", fal=True, fal_type='low')
    atm = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name="XGB", fal=True,fal_type='medium')
    ath = get_results(data_name, fudan_filepath, 'all', select_features=True, clf_name="XGB", fal=True, fal_type='high')

    # ALL XGB ___ ALL FEATURES ___
    aff = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name="XGB", fal=False,fal_type='high')
    afl = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name="XGB", fal=True, fal_type='low')
    afm = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name="XGB", fal=True,fal_type='medium')
    afh = get_results(data_name, fudan_filepath, 'all', select_features=False, clf_name="XGB", fal=True,fal_type='high')
    xgb_res = pd.concat([otf, otl, otm, oth, off, ofl, ofm, ofh, ytf, ytl, ytm, yth, yff, yfl, yfm, yfh, atf, atl, atm, ath, aff, afl,afm, afh])
    if not os.path.exists(str(Config.LOG_DIR) + "/" + FUDAN + "/final_results/"):
        os.makedirs(os.path.join(Config.LOG_DIR, FUDAN, 'final_results'))
    xgb_res.to_csv(os.path.join(Config.LOG_DIR, FUDAN, 'final_results/xgb_final_results.csv'))
    return xgb_res

rf_res = rf_results()
#svm_res = svm_results()
#xgb_res = xgb_results()
#knn_res = knn_results()

result_mega_table = pd.concat([rf_res, svm_res, xgb_res, knn_res])
if not os.path.exists(str(Config.LOG_DIR) + "/" + FUDAN + "/final_results/"):
    os.makedirs(os.path.join(Config.LOG_DIR, FUDAN, 'final_results'))
result_mega_table.to_csv(os.path.join(Config.LOG_DIR, FUDAN, 'final_results/final_results.csv'))