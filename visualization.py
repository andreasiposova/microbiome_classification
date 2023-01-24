from statistics import mean
import os
import pandas as pd

from utils import Config

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix




def get_rf_scores_params(test_scores):
    scores = []
    # Extract the accuracy values
    accuracy_values = [score['accuracy'] for score in test_scores]
    precision_values = [score['precision'] for score in test_scores]
    recall_values = [score['recall'] for score in test_scores]
    roc_auc_values = [score['roc_auc'] for score in test_scores]
    f1_values = [score['f1'] for score in test_scores]
    f2_values = [score['f2'] for score in test_scores]
    scores.append(accuracy_values)
    scores.append(precision_values)
    scores.append(recall_values)
    scores.append(roc_auc_values)
    scores.append(f1_values)
    scores.append(f2_values)

    param_values = [score['params'] for score in test_scores]
    param_combinations = []
    for i in param_values:
        p = ""
        for key, value in i.items():
            p += f"{key}: {value}\n"
        param_combinations += [p]

    df = pd.DataFrame(test_scores)
    params_df = pd.concat([df.drop(['params'], axis=1), df['params'].apply(pd.Series)], axis=1)
    n_estimators_mean_metrics = params_df[
        ['n_estimators', 'accuracy', 'precision', 'recall', 'roc_auc', 'f1', 'f2']].groupby('n_estimators').mean()
    n_estimators_mean_metrics.reset_index(inplace=True)
    max_depth_mean_metrics = params_df[['max_depth', 'accuracy', 'precision', 'recall', 'roc_auc', 'f1', 'f2']].groupby(
        'max_depth').mean()
    max_depth_mean_metrics.reset_index(inplace=True)
    max_features_mean_metrics = params_df[
        ['max_features', 'accuracy', 'precision', 'recall', 'roc_auc', 'f1', 'f2']].groupby('max_features').mean()
    max_features_mean_metrics.reset_index(inplace=True)
    min_samples_leaf_mean_metrics = params_df[
        ['min_samples_leaf', 'accuracy', 'precision', 'roc_auc', 'recall', 'f1', 'f2']].groupby(
        'min_samples_leaf').mean()
    min_samples_leaf_mean_metrics.reset_index(inplace=True)
    min_samples_split_mean_metrics = params_df[
        ['min_samples_split', 'accuracy', 'precision', 'roc_auc', 'recall', 'f1', 'f2']].groupby(
        'min_samples_split').mean()
    min_samples_split_mean_metrics.reset_index(inplace=True)

    return scores, param_combinations, n_estimators_mean_metrics, max_depth_mean_metrics, max_features_mean_metrics, min_samples_split_mean_metrics, min_samples_leaf_mean_metrics


def grid_search_plot(hyperparam, scores, data_name, file_name):
    accuracy_values = scores[0]
    precision_values = scores[1]
    recall_values = scores[2]
    roc_auc_values = scores[3]
    f1_values = scores[4]
    f2_values = scores[5]

    if not os.path.exists(str(Config.PLOTS_DIR) + "/" + str(data_name) + "/" + str(file_name)):
        os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, file_name))

    plt.figure(figsize=(7, 5))
    plt.plot(hyperparam, accuracy_values, label='Accuracy')
    plt.plot(hyperparam, precision_values, label='Precision')
    plt.plot(hyperparam, recall_values, label='Recall')
    plt.plot(hyperparam, roc_auc_values, label='ROC AUC Score')
    plt.plot(hyperparam, f1_values, label='F1 Score')
    plt.plot(hyperparam, f2_values, label='F2 Score')
    plt.xticks(rotation=45, ha='right', fontsize=2)
    # plt.xlabel(hyperparam_string)
    plt.ylabel('Score')
    plt.legend()
    plt.title(file_name)
    plt.tight_layout()
    plt.savefig(os.path.join(Config.PLOTS_DIR, data_name, file_name, f"gridsearch_rf.png"))
    #plt.show()


def sensitivity_plot(hp_mean_metrics, data_name, file_name):
    columns = hp_mean_metrics.columns
    columns_list = [col for col in columns]
    hyperparam_name = columns_list[0]
    scores = []
    for i in columns_list:
        score = hp_mean_metrics[i].to_list()
        scores.append(score)
    # values = hp_mean_metrics.Series.values.to_list()
    # values_list = [val for val in values]

    if not os.path.exists(str(Config.PLOTS_DIR) + "/" + str(data_name) + "/" + str(file_name)):
        os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, file_name))

    plt.figure(figsize=(7, 5))
    plt.plot(scores[0], scores[1], label='Accuracy')
    plt.plot(scores[0], scores[2], label='Precision')
    plt.plot(scores[0], scores[3], label='Recall')
    plt.plot(scores[0], scores[4], label='ROC AUC Score')
    plt.plot(scores[0], scores[5], label='F1 Score')
    plt.plot(scores[0], scores[6], label='F2 Score')
    # plt.xticks(rotation=45, ha='right', fontsize=2)
    plt.xlabel(hyperparam_name)
    plt.ylabel('Score')
    plt.legend()
    plt.title(file_name)
    plt.tight_layout()
    plt.savefig(
        os.path.join(Config.PLOTS_DIR, data_name, file_name, f"{hyperparam_name}_rf_sensitivity_gridsearch_avg.png"))
    #plt.show()


def cm_plot(y_test, y_pred, data_name, file_name, test_or_val):
    if not os.path.exists(str(Config.PLOTS_DIR) + "/" + str(data_name) + "/" + str(file_name)):
        os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, file_name))

    label_mapping = {'healthy': 0, 'CRC': 1}
    reverse_mapping = {value: key for key, value in label_mapping.items()}
    y_test = [reverse_mapping[label] for label in y_test]
    y_pred = [reverse_mapping[label] for label in y_pred]
    labels = ['healthy', 'CRC']
    cm = confusion_matrix(y_test, y_pred, labels=labels)
    plt.figure(figsize=(7, 5))
    disp = ConfusionMatrixDisplay(cm, display_labels=labels)
    disp.plot()
    plt.title(file_name)
    plt.savefig(os.path.join(Config.PLOTS_DIR, data_name, file_name, f"rf_best_estimator_{test_or_val}_cm.png"))
    #plt.show()
