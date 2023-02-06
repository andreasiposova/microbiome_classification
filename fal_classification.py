import json
import os
import ast
import numpy as np
from math import sqrt

import forestci as fci
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, fbeta_score
from sklearn.preprocessing import MinMaxScaler

from data_loading import load_tsv_files
from feature_selection import select_features_from_paper
from preprocessing import preprocess_data, preprocess_huadong, full_preprocessing_y_o_labels, \
    apply_feature_abundance_limits
from utils import Config

import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

from visualization import plot_conf_int

FUDAN = 'fudan'
HUADONG1 = 'huadong1'
HUADONG2 = 'huadong2'

file_names = list(("pielou_e_diversity", "simpson_diversity", "phylum_relative", "observed_otus_diversity", "family_relative", "class_relative", "fb_ratio", "enterotype", "genus_relative", "species_relative", "shannon_diversity", "domain_relative", "order_relative", "simpson_e_diversity"))

yang_metadata_path = "data/Yang_PRJNA763023/metadata.csv"
fudan_filepath = 'data/Yang_PRJNA763023/Yang_PRJNA763023_SE/parsed/normalized_results/'
huadong_filepath_1 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_1/parsed/normalized_results'
huadong_filepath_2 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_2/parsed/normalized_results'
young_old_labels_path = 'data/Yang_PRJNA763023/SraRunTable.csv'



def get_best_params(data_name, group, file_name, clf, param_file = 'best_params'):
    with open(os.path.join(Config.LOG_DIR, data_name, group, file_name, f"{clf}_{param_file}.txt"), "rb") as f:
        params = f.read()
        params = ast.literal_eval(params.decode())
        params = params['params']
    return params



def load_preprocessed_data(data_name=FUDAN, filepath=fudan_filepath, group='old', select_features = True):
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
                X_train_1, X_test_1, y_train, y_test = preprocess_data(data[key], yang_metadata_path)
                X_h1, y_h1 = preprocess_huadong(huadong_data1[key], yang_metadata_path)
                X_h2, y_h2 = preprocess_huadong(huadong_data2[key], yang_metadata_path)
            else:
                X_train_1, X_test_1, X_val_1, y_train, y_test, y_val = full_preprocessing_y_o_labels(data, huadong_data1, huadong_data2, key, yang_metadata_path, young_old_labels_path, group)
                if select_features == False:
                    file_name = "all_features"
                    print(f"Running experiments on {group} samples, without feature selection")

                if select_features == True:
                    file_name = "selected_features"
                    print(f"Running experiments on {group} samples, with feature selection")
                    # top_features = calculate_feature_importance(X_train_1, y_train, group)
                    # top_features_names = list(map(lambda x: x[0], top_features))
                    # print(top_features_names)
                    # X_train_1 = X_train_1[top_features_names]
                    # X_train.to_csv('data/selected_features_old.csv')
                    # common_cols_f = set(X_test_1.columns).intersection(X_train_1.columns)
                    # common_cols_fv = set(X_val_1.columns).intersection(X_train_1.columns)
                    # X_test_1 = X_test_1[common_cols_f]
                    # X_val_1 = X_val_[common_cols_fv]
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
    X_train = X_train.append(X_test)
    y_train = y_train + y_test
    print("number of samples in training set: ", len(X_train))
    print("number of samples in test set: ", len(X_test))
    print("number of samples in validation set: ", len(X_val))




    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)

    return X_train, X_test, X_val, y_train, y_test, y_val

def perform_rf_classification(X_train, X_test, X_val, y_train, y_test, y_val, params, group, file_name):

    results_df = pd.DataFrame()


    clf = RandomForestClassifier(**params)
    #clf.fit(X_train, y_train)
    #y_train_pred = clf.predict(X_train)
    #y_test_pred = clf.predict(X_test)
    #y_val_pred = clf.predict(X_val)
    threshold = 0.515
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
    y_train_pred = (y_train_prob[:, 1] >= threshold).astype('int')
    plot_conf_int(y_train, y_train_prob, X_train, X_train, clf, data_name=FUDAN, file_name=file_name, group=group, set_name="train")


    #y_test_prob = clf.predict_proba(X_test)
    #y_test_pred = (y_test_prob[:, 1] >= threshold).astype('int')
    #plot_conf_int(y_test, y_test_prob, X_train, X_test, clf, data_name=FUDAN, file_name=file_name, group=group, set_name="test")

    y_val_prob = clf.predict_proba(X_val)
    y_val_pred = (y_val_prob[:, 1] >= threshold).astype('int')
    plot_conf_int(y_val, y_val_prob, X_train, X_val, clf, data_name=FUDAN, file_name=file_name, group=group, set_name="val")

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
    train_set_res = {'accuracy': acc_train, 'precision': prec_train, 'recall': recall_train, 'roc_auc': roc_auc_train, 'f1': f1_train,
                          'f2': f2_train, 'auc_conf_int': ci_train}
    train_results = train_results.append(train_set_res, ignore_index=True)


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
    val_set_res = {'accuracy': acc_val, 'precision': prec_val, 'recall': recall_val,
                          'roc_auc': roc_auc_val, 'f1': f1_val,
                          'f2': f2_val, 'auc_conf_int': ci_val}
    val_results = val_results.append(val_set_res, ignore_index=True)

    train_results = train_results.mean()
    #test_results = test_results.mean()
    val_results = val_results.mean()

    results_df = results_df.append(train_results, ignore_index=True)
    #results_df = results_df.append(test_results, ignore_index=True)
    results_df = results_df.append(val_results, ignore_index=True)

    return results_df

def perform_classification():

    results_df = pd.DataFrame()

    clf = RandomForestClassifier(**params)
    #clf.fit(X_train, y_train)
    #y_train_pred = clf.predict(X_train)
    #y_test_pred = clf.predict(X_test)
    #y_val_pred = clf.predict(X_val)
    threshold = 0.515
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
    y_train_pred = (y_train_prob[:, 1] >= threshold).astype('int')



    #y_test_prob = clf.predict_proba(X_test)
    #y_test_pred = (y_test_prob[:, 1] >= threshold).astype('int')

    y_val_prob = clf.predict_proba(X_val)
    y_val_pred = (y_val_prob[:, 1] >= threshold).astype('int')


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
    train_set_res = {'accuracy': acc_train, 'precision': prec_train, 'recall': recall_train, 'roc_auc': roc_auc_train, 'f1': f1_train,
                          'f2': f2_train, 'auc_conf_int': ci_train}
    train_results = train_results.append(train_set_res, ignore_index=True)


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
    val_set_res = {'accuracy': acc_val, 'precision': prec_val, 'recall': recall_val,
                          'roc_auc': roc_auc_val, 'f1': f1_val,
                          'f2': f2_val, 'auc_conf_int': ci_val}
    val_results = val_results.append(val_set_res, ignore_index=True)

    train_results = train_results.mean()
    #test_results = test_results.mean()
    val_results = val_results.mean()

    results_df = results_df.append(train_results, ignore_index=True)
    #results_df = results_df.append(test_results, ignore_index=True)
    results_df = results_df.append(val_results, ignore_index=True)

    return results_df




def get_results(data_name, filepath, group, select_features, clf, fal):
    X_train, X_test, X_val, y_train, y_test, y_val = load_preprocessed_data(FUDAN, fudan_filepath, group='old', select_features=True)
    if select_features == True:
        params = get_best_params(FUDAN, group, 'selected_features', clf, param_file='best_params')
        if fal == True:
            X_train = apply_feature_abundance_limits(X_train)
            X_test = apply_feature_abundance_limits(X_test)
            X_val = apply_feature_abundance_limits(X_val)
            if clf == "RF":
                results = perform_rf_classification(X_train, X_test, X_val, y_train, y_test, y_val, params, group=group, file_name="selected_features")
            else:
                results = perform_classification(X_train, X_test, X_val, y_train, y_test, y_val, params, group=group, file_name="selected_features")
            print(results)
        if fal == False:
            if clf == "RF":
                results = perform_rf_classification(X_train, X_test, X_val, y_train, y_test, y_val, params, group=group, file_name="selected_features")
            else:
                results = perform_classification(X_train, X_test, X_val, y_train, y_test, y_val, params, group=group, file_name="selected_features")

    if select_features == False:
        params = get_best_params(FUDAN, group, 'all_features', clf,  param_file='best_params')
        if fal == True:
            X_train = apply_feature_abundance_limits(X_train)
            X_test = apply_feature_abundance_limits(X_test)
            X_val = apply_feature_abundance_limits(X_val)
            if clf == "RF":
                results = perform_rf_classification(X_train, X_test, X_val, y_train, y_test, y_val, params,
                                                    group=group, file_name="selected_features")
            else:
                results = perform_classification(X_train, X_test, X_val, y_train, y_test, y_val, params,
                                                 group=group, file_name="selected_features")
            print(results)
        if fal == False:
            if clf == "RF":
                results = perform_rf_classification(X_train, X_test, X_val, y_train, y_test, y_val, params,
                                                    group=group, file_name="selected_features")
            else:
                results = perform_classification(X_train, X_test, X_val, y_train, y_test, y_val, params,
                                                 group=group, file_name="selected_features")

            print(results)





get_rf_results(FUDAN, fudan_filepath, 'old', select_features=True, clf = "RF", fal=True)