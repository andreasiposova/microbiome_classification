from statistics import mean
import os
import numpy as np
import pandas as pd
from scipy.stats import stats

from utils import Config

import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix
import forestci as fci




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
    class_weight_mean_metrics = params_df[
        ['class_weight', 'accuracy', 'precision', 'roc_auc', 'recall', 'f1', 'f2']].groupby(
        'class_weight').mean()
    class_weight_mean_metrics.reset_index(inplace=True)

    return scores, param_combinations, n_estimators_mean_metrics, max_depth_mean_metrics, max_features_mean_metrics, min_samples_leaf_mean_metrics, min_samples_split_mean_metrics, class_weight_mean_metrics


def grid_search_plot(hyperparam, scores, data_name, group, file_name):
    accuracy_values = scores[0]
    precision_values = scores[1]
    recall_values = scores[2]
    roc_auc_values = scores[3]
    f1_values = scores[4]
    f2_values = scores[5]

    if not os.path.exists(str(Config.PLOTS_DIR) + "/" + str(data_name) + "/" + str(group) +"/"+ str(file_name)):
        os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, group, file_name))

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


def sensitivity_plot(hp_mean_metrics, data_name, group, file_name):
    columns = hp_mean_metrics.columns
    columns_list = [col for col in columns]
    hyperparam_name = columns_list[0]
    scores = []
    for i in columns_list:
        score = hp_mean_metrics[i].to_list()
        scores.append(score)
    # values = hp_mean_metrics.Series.values.to_list()
    # values_list = [val for val in values]

    if not os.path.exists(str(Config.PLOTS_DIR) + "/" + str(data_name) + "/" + group + "/" + str(file_name)):
        os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, group, file_name))

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


def cm_plot(y_test, y_pred, data_name, group, file_name, test_or_val, clf_name, final, fal, fal_type):
    if final==True:
        if not os.path.exists(str(Config.PLOTS_DIR) + "/" + str(data_name) + "/" + "final_results/" + group + "/" + str(file_name)):
            os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, "final_results", group, file_name))

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
        if fal==True:
            plt.savefig(os.path.join(Config.PLOTS_DIR, data_name, "final_results", group, file_name,
                                     f"{clf_name}_best_estimator_{test_or_val}_fal_{fal_type}_cm.png"))
        if fal==False:
            plt.savefig(os.path.join(Config.PLOTS_DIR, data_name, "final_results", group, file_name,
                                     f"{clf_name}_best_estimator_{test_or_val}_cm.png"))
        plt.close()
    else:

        if not os.path.exists(str(Config.PLOTS_DIR) + "/" + str(data_name) + "/" + group + "/" + str(file_name)):
            os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, group, file_name))

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
        plt.savefig(os.path.join(Config.PLOTS_DIR, data_name, group, file_name, f"{clf_name}_best_estimator_{test_or_val}_cm.png"))
        plt.close()

def grid_search_train_test_plot(train_scores, test_scores, data_name, group, file_name, clf_name):
    # plot the train and test scores
    if not os.path.exists(str(Config.PLOTS_DIR) + "/" + str(data_name) + "/" + group + "/" + file_name):
        os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, group, file_name))
    plt.plot(train_scores, label='train score')
    plt.plot(test_scores, label='test score')
    plt.title('GridSearch CV Train vs. Test score')
    plt.legend()
    plt.savefig(os.path.join(Config.PLOTS_DIR, data_name, group, file_name, f"{clf_name}_grid_search_scores.png"))
    plt.close()


def create_scores_dataframe(grid_clf, param_name, num_results=15, negative=True, graph=True, display_all_params=True):
    clf = grid_clf.best_estimator_
    clf_params = grid_clf.best_params_
    if negative:
        clf_score = -grid_clf.best_score_
    else:
        clf_score = grid_clf.best_score_
    clf_stdev = grid_clf.cv_results_['std_test_score'][grid_clf.best_index_]
    cv_results = grid_clf.cv_results_

    # pick out the best results
    # =========================
    scores_df = pd.DataFrame(cv_results).sort_values(by='rank_test_score')


"""def plot_hyperparam_sensitivity(param, param_ranges, acc, prec, rec, roc_auc, f1, data_name, group, name):
    plt.figure(figsize=(7, 5))
    x_str = [str(value) for value in param_ranges[param]]
    if "None" in x_str or "True" in x_str or "False" in x_str:
        x = x_str
    else:
        x = param_ranges[param]
    plt.plot(x, metric, label="Accuracy")
    plt.plot(x, metric, label="Precision")
    plt.plot(x, metric, label="Recall")
    plt.plot(x, roc_auc, label="ROC AUC")
    plt.plot(x, f1, label="F1")
    # plt.plot(x, f2, label="F2")
    plt.xlabel(param)
    plt.ylabel("score")
    plt.legend()
    plt.tight_layout()
    plt.savefig(
        os.path.join(os.path.join(Config.PLOTS_DIR, data_name, f"sensitivity/{param}_rf_sensitivity_test.png")))
    plt.close()
"""
def plot_conf_int(y_true, y_pred, X_train, X_pred, clf, data_name, file_name, group, fal, fal_type, set_name):

    if not os.path.exists(str(Config.PLOTS_DIR) + "/" + str(data_name) + "/final_results/" + str(group) + "/" + str(file_name) + "/RF"):
        os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, "final_results", group, file_name, "RF"))
    y_true = np.array(y_true)
    idx_crc = np.where(y_true == 1)[0]
    idx_healthy = np.where(y_true == 0)[0]

    fig, ax = plt.subplots(1)
    ax.hist(y_pred[idx_crc, 1], histtype='step', label='CRC', color='orange')
    ax.hist(y_pred[idx_healthy, 1], histtype='step', label='healthy', color='blue')
    ax.set_xlabel('Prediction (CRC probability)')
    ax.set_ylabel('Number of observations')
    plt.legend()
    if fal == True:
        plt.savefig(os.path.join(Config.PLOTS_DIR, data_name, "final_results", group, file_name, "RF", f"RF_{set_name}_histogram_CI_fal_{fal_type}.png"))
    if fal == False:
        plt.savefig(os.path.join(Config.PLOTS_DIR, data_name, "final_results", group, file_name, "RF", f"RF_{set_name}_histogram_CI.png"))
    plt.close()

    # Calculate the variance
    spam_V_IJ_unbiased = fci.random_forest_error(clf, X_train, X_pred)

    # Plot forest prediction for emails and standard deviation for estimates
    # Blue points are spam emails; Green points are non-spam emails
    fig, ax = plt.subplots(1)
    ax.scatter(y_pred[idx_crc, 1],
               np.sqrt(spam_V_IJ_unbiased[idx_crc]),
               label='CRC', color='orange')

    ax.scatter(y_pred[idx_healthy, 1],
               np.sqrt(spam_V_IJ_unbiased[idx_healthy]),
               label='healthy', color='blue')

    ax.set_xlabel('Prediction (CRC probability)')
    ax.set_ylabel('Standard deviation')
    plt.legend()
    if fal==True:
        plt.savefig(os.path.join(Config.PLOTS_DIR, data_name, "final_results", group, file_name, "RF", f"RF_{set_name}_fal_{fal_type}_scatterplot_CI.png"))
    if fal == False:
        plt.savefig(os.path.join(Config.PLOTS_DIR, data_name, "final_results", group, file_name, "RF", f"RF_{set_name}_scatterplot_CI.png"))
    plt.close()

def prob_boxplot(y_true, probs, data_name, group, file_name, set_name, fal, fal_type):
    # Split the predicted probabilities into positive and negative groups based on the true labels
    crc = probs[:, -1]
    healthy = probs[:,0:1]
    probs = pd.DataFrame(probs)

    vals, names, xs = [], [], []
    for i, col in enumerate(probs.columns):
        vals.append(probs[col].values)
        names.append(col)
        xs.append(
            np.random.normal(i + 1, 0.08, probs[col].values.shape[0]))  # adds jitter to the data points - can be adjusted

    plt.boxplot(vals, labels=names)
    palette = [ '#84cdfc' , '#fdb88e']
    for x, val, c in zip(xs, vals, palette):
        plt.scatter(x, val, alpha=0.2, color=c)
    plt.xlabel("Class", fontweight='normal', fontsize=12)
    plt.ylabel("Predicted POD", fontweight='normal', fontsize=12)
    #plt.axhline(y=0.5, color='#b9b9b9', linestyle='--', linewidth=1, label='Threshold Value')


    """
    #crc = [probs[i, 1] for i in range(len(y_true)) if y_true[i]==1]
    #healthy = [probs[i, 1] for i in range(len(y_true)) if y_true[i] == 0]
    #t_statistic, p_value = stats.ttest_ind(crc, healthy)
    differences = np.array(crc) - np.array(healthy)
    t, p = stats.ttest_rel(crc, healthy)
    # Plot the results



    # Plot the positive and negative groups as boxplots
    fig, ax = plt.subplots(1, 2, figsize=(5, 7), sharey=True)

    # Create violin plots for positive and negative groups
    #ax[0].violinplot(crc, showmeans=True, showmedians=False, showextrema=True)
    #ax[1].violinplot(healthy, showmeans=True, showmedians=False, showextrema=True)

    bplot1 = ax.boxplot(crc, positions=[1], widths=0.6)
    bplot2 = ax.boxplot(healthy, positions=[2], widths=0.6)

    # Add labels and title
    # Add labels and title
    ax.set_xlabel('Class')
    ax.set_ylabel('Predicted POD')
    ax.set_title('Predicted POD for CRC vs. healthy group')
    ax.set_xticklabels(['CRC', 'Healthy'])

    # Set y-limits for both subplots
    ax[0].set_ylim([0, 1])
    ax[1].set_ylim([0, 1])
    yticks = np.arange(0, 1.1, 0.2)
    ax[0].set_yticks(yticks)
    ax[1].set_yticks(yticks)
    x_ticks = np.arange(0, 1.1, 1)
    ax[0].set_xticks(x_ticks)
    ax[1].set_xticks(x_ticks)

    # Add labels and title
    ax[0].set_title('Positive Group')
    ax[1].set_title('Negative Group')
    ax[0].set_xlabel('Probability')
    ax[1].set_xlabel('Probability')
    """
    if not os.path.exists(os.path.join(Config.PLOTS_DIR, data_name, "final_results", group, file_name, "RF")):
        os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, "final_results", group, file_name, "RF"))
    if fal==True:
        plt.savefig(os.path.join(Config.PLOTS_DIR, data_name, "final_results", group, file_name, "RF",
                                 f"RF_{set_name}_fal_{fal_type}_boxplots.png"))
    if fal==False:
        plt.savefig(os.path.join(Config.PLOTS_DIR, data_name, "final_results", group, file_name, "RF",
                                 f"RF_{set_name}_boxplots.png"))
    plt.close()

def plot_prob_histogram(y_true, y_prob, clf_name, bins, data_name, group, file_name, set_name, fal, fal_type):
    # Extract predict_proba results for samples with y_pred label 0
    probs = y_prob[:,1]

    probs = probs.tolist()
    probs_healthy = [probs[i] for i, x in enumerate(y_true) if x == 0]
    probs_CRC = [probs[i] for i, x in enumerate(y_true) if x == 1]

    # Plot histograms for each y_pred label
    plt.hist(probs_healthy, bins=bins, alpha=0.5, histtype='step', label='healthy', color='blue')
    plt.hist(probs_CRC, bins=bins, alpha=0.5, histtype='step', label='CRC', color='orange')
    plt.legend()
    if not os.path.exists(os.path.join(Config.PLOTS_DIR, data_name, "final_results", group, file_name, clf_name)):
        os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, "Final_results", group, file_name, clf_name))
    if fal==True:
        plt.savefig(os.path.join(Config.PLOTS_DIR, data_name, "final_results", group, file_name, clf_name, f"{clf_name}_{set_name}_fal_{fal_type}_histogram.png"))
    else:
        plt.savefig(os.path.join(Config.PLOTS_DIR, data_name, "final_results", group, file_name, clf_name,
                                 f"{clf_name}_{set_name}_histogram.png"))
    plt.close()