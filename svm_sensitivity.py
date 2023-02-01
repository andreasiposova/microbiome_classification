import argparse
import os
import pandas as pd
import numpy as np
from sklearn import svm
from sklearn.preprocessing import MinMaxScaler

from feature_selection import select_features_from_paper
from utils import setup_logging, Config
from preprocessing import preprocess_data, preprocess_huadong, full_preprocessing_y_o_labels

from datetime import datetime
from data_loading import load_tsv_files, load_young_old_labels

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, \
    fbeta_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate
from sklearn.base import clone
from sklearn.pipeline import Pipeline

import matplotlib

from visualization import plot_hyperparam_sensitivity

matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix

# Set up logging
# logfile = setup_logging("tune_random_forest") # logger

# Set up logging
logger = setup_logging("tune_random_forest")
log_file = "rf" + "_" + datetime.now().strftime("%Y%m%d_%H%M%S")

FUDAN = 'fudan'
HUADONG1 = 'huadong1'
HUADONG2 = 'huadong2'

file_names = list(
    ("pielou_e_diversity", "simpson_diversity", "phylum_relative", "observed_otus_diversity", "family_relative",
     "class_relative", "fb_ratio", "enterotype", "genus_relative", "species_relative", "shannon_diversity",
     "domain_relative",
     "order_relative", "simpson_e_diversity"))

yang_metadata_path = "data/Yang_PRJNA763023/metadata.csv"
fudan_filepath = 'data/Yang_PRJNA763023/Yang_PRJNA763023_SE/parsed/normalized_results/'
huadong_filepath_1 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_1/parsed/normalized_results'
huadong_filepath_2 = 'data/Yang_PRJNA763023/Yang_PRJNA763023_PE_2/parsed/normalized_results'
young_old_labels_path = 'data/Yang_PRJNA763023/SraRunTable.csv'

def calculate_performance(y_true, y_pred):
    accuracies = []
    precisions = []
    recalls = []
    roc_aucs = []
    f1s = []
    f2s = []

    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    roc_auc = roc_auc_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    f2 = fbeta_score(y_true, y_pred, beta=2)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    roc_aucs.append(roc_auc)
    f1s.append(f1)
    f2s.append(f2)

    return accuracies, precisions, recalls, roc_aucs, f1s, f2s

def sensitivity_analysis(data_name, filepath, group, select_features = True):
    full_results = []
    y_o_labels = load_young_old_labels(young_old_labels_path)
    data = load_tsv_files(filepath)
    huadong_data1 = load_tsv_files(huadong_filepath_1)
    huadong_data2 = load_tsv_files(huadong_filepath_2)

    X_train = pd.DataFrame()
    X_test = pd.DataFrame()
    X_val = pd.DataFrame()

    for key in data:
        if key == "genus_relative" or key == "family_relative":
            # X_train, X_test, y_train, y_test = preprocess_data(data[key], yang_metadata_path) #preprocess_fudan_data?
            if group == "all":
                X_train_1, X_test_1, y_train, y_test = preprocess_data(data[key], yang_metadata_path)
                X_h1, y_h1 = preprocess_huadong(huadong_data1[key], yang_metadata_path)
                X_h2, y_h2 = preprocess_huadong(huadong_data2[key], yang_metadata_path)
            else:
                X_train_1, X_test_1, X_val_1, y_train, y_test, y_val = full_preprocessing_y_o_labels(data, huadong_data1,
                                                                                                     huadong_data2, key,
                                                                                                     yang_metadata_path,
                                                                                                     young_old_labels_path,
                                                                                                     group)

            X_train = pd.concat([X_train, X_train_1], axis=1)
            X_test = pd.concat([X_test, X_test_1], axis=1)
            X_val = pd.concat([X_val, X_val_1], axis=1)

    common_cols_t = set(X_test.columns).intersection(X_val.columns)
    common_cols_v = set(X_val.columns).intersection(X_test.columns)

    # filling missing values in huadong cohort with zeros
    # as two files are concatenated for huadong cohort files
    # they contain columns that are not compatible
    # thus creating missing values - they are replaced with 0 as it means the abundace of that bacteria is anyway 0
    X_val = X_val.fillna(0)
    X_val = X_val[common_cols_v]
    X_train = X_train[common_cols_t]
    X_test = X_test[common_cols_t]
    # X_train = X_train.append(X_test)
    # y_train = y_train + y_test



    if select_features == False:
        file_name = "all_features"

    if select_features == True:
        file_name = "selected_features"
        #top_features = calculate_feature_importance(X_train, y_train, group)
        #top_features_names = list(map(lambda x: x[0], top_features))
        #print(top_features_names)
        #X_train = X_train[top_features_names]
        #X_train.to_csv('data/selected_features_old.csv')
        #common_cols_f = set(X_test.columns).intersection(X_train.columns)
        #common_cols_fv = set(X_val.columns).intersection(X_train.columns)
        #X_test = X_test[common_cols_f]
        #X_val = X_val[common_cols_fv]
        X_test = select_features_from_paper(X_test, key, group)
        X_train = select_features_from_paper(X_train, key, group)
        X_val = select_features_from_paper(X_val, key, group)

    scaler = MinMaxScaler()
    scaler.fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)
    X_val = scaler.transform(X_val)


    # define estimator
    model = svm.SVC()
    estimator = Pipeline([("model", model)])
    # define a range of values for the maximum depth
    param_ranges = {'C':[1,10],# 100, 1000],
                    'gamma': [1, 0.5, 0.2, 0.1, 0.01, 0.001,0.0001],
                    'kernel': ['linear', 'rbf'],
                    'random_state': [1234]
    }

    # initialize an empty list to store the performance scores
    scores = []

    # iterate over the range of values for maximum depth
    for param in param_ranges:
        print(f"Running sensitivity analysis on parameter {param}")
        accuracies_cv = []
        precisions_cv = []
        recalls_cv = []
        roc_aucs_cv = []
        f1s_cv = []
        f2s_cv = []

        roc_aucs_train = []
        roc_aucs_test = []
        roc_aucs_val = []



        for value in param_ranges[param]:
            #print(f" value:  {value}")
            model = clone(estimator)  # clone(estimator)
            model.set_params(**{f"model__{param}": value})
            cv = RepeatedStratifiedKFold(n_splits=2, random_state=1234)
            scores = cross_validate(model, X_train, y_train,
                                    scoring=["accuracy", "precision", "recall", "roc_auc", "f1"], cv=cv, n_jobs=-1,
                                    return_estimator=True)

            # holdout
            #print(model.get_params())
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)
            y_pred_val = model.predict(X_val)
            acc_train, prec_train, rec_train, roc_auc_train, f1_train, f2_train = calculate_performance(y_train, y_pred_train)
            acc_test, prec_test, rec_test, roc_auc_test, f1_test, f2_test = calculate_performance(y_test, y_pred_test)
            acc_val, prec_val, rec_val, roc_auc_val, f1_val, f2_val = calculate_performance(y_val, y_pred_val)
            roc_aucs_train.append(roc_auc_train)
            roc_aucs_test.append(roc_auc_test)
            roc_aucs_val.append(roc_auc_val)

            # cross validation results averaged
            accuracies_cv.append(scores["test_accuracy"].mean())
            precisions_cv.append(scores["test_precision"].mean())
            recalls_cv.append(scores["test_recall"].mean())
            roc_aucs_cv.append(scores["test_roc_auc"].mean())
            f1s_cv.append(scores["test_f1"].mean())
            #f2s_cv.append(scores["test_f2"].mean())


        if not os.path.exists(str(Config.PLOTS_DIR) +"/"+ str(data_name) +"/"+ str(group) +"/"+ "all_features/"):
            os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, group, f"all_features"))

        plt.figure(figsize=(7, 5))
        x_str = [str(value) for value in param_ranges[param]]
        if "None" in x_str or "True" in x_str or "False" in x_str:
            x = x_str
        else:
            x = param_ranges[param]
        plt.plot(x, accuracies_cv, label="Accuracy")
        plt.plot(x, precisions_cv, label="Precision")
        plt.plot(x, recalls_cv, label="Recall")
        plt.plot(x, roc_aucs_cv, label="ROC AUC")
        plt.plot(x, f1s_cv, label="F1")
        # plt.plot(x, f2, label="F2")
        plt.xlabel(param)
        plt.ylabel("score")
        plt.legend()
        plt.title(f"Cross-validated {param} sensitivity results")
        plt.tight_layout()
        if select_features == True:
            if not os.path.exists(
                    str(Config.PLOTS_DIR) + "/" + str(data_name) + "/" + str(group) + "/" + "selected_features/"):
                os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, group, f"selected_features"))
            plt.savefig(
                os.path.join(
                    os.path.join(Config.PLOTS_DIR, data_name, group, f"selected_features/SVM_{param}_sensitivity_test.png")))
        if select_features == False:
            if not os.path.exists(
                    str(Config.PLOTS_DIR) + "/" + str(data_name) + "/" + str(group) + "/" + "all_features/"):
                os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, group, f"all_features"))
            plt.savefig(
                os.path.join(
                    os.path.join(Config.PLOTS_DIR, data_name, group,
                                 f"all_features/SVM_{param}_sensitivity_test.png")))


        plt.figure(figsize=(7, 5))
        x_str = [str(value) for value in param_ranges[param]]
        if "None" in x_str or "True" in x_str or "False" in x_str:
            x = x_str
        else:
            x = param_ranges[param]
        plt.plot(x, roc_aucs_train, label="Train")
        plt.plot(x, roc_aucs_test, label="Test")
        plt.plot(x, roc_aucs_val, label="Validation")
        plt.plot(x, roc_aucs_cv, label="CV mean")
        plt.xlabel(param)
        plt.ylabel("score")
        plt.legend()
        plt.title(f"ROC AUC comparison - {param}")
        plt.tight_layout()
        if select_features == True:
            if not os.path.exists(str(Config.PLOTS_DIR) + "/" + str(data_name) + "/" + str(group) + "/" + "selected_features/"):
                os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, group, f"selected_features"))
            plt.savefig(
                os.path.join(
                    os.path.join(Config.PLOTS_DIR, data_name, group, f"selected_features/SVM_{param}_roc_auc.png")))
        if select_features == False:
            if not os.path.exists(
                    str(Config.PLOTS_DIR) + "/" + str(data_name) + "/" + str(group) + "/" + "all_features/"):
                os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, group, f"all_features"))
            plt.savefig(
                os.path.join(
                    os.path.join(Config.PLOTS_DIR, data_name, group, f"all_features/SVM_{param}_roc_auc.png")))

sensitivity_analysis(FUDAN, fudan_filepath, group="young", select_features=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=FUDAN)
    parser.add_argument('--filepath', type=str, default=fudan_filepath)
    parser.add_argument('--group', type=str, default="old")

    args = parser.parse_args()
    data_name = args.data_name
    if data_name == FUDAN:
        sensitivity_analysis(data_name=args.data_name, filepath=args.filepath, group = "young")

    else:
        raise ValueError()
