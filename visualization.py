import matplotlib.pyplot as plt
import os
from utils import Config

def get_scores(test_scores):
    scores = []
    # Extract the accuracy values
    accuracy_values = [score['accuracy'] for score in test_scores]
    precision_values = [score['precision'] for score in test_scores]
    recall_values = [score['recall'] for score in test_scores]
    roc_auc_values = [score['roc_auc'] for score in test_scores]
    f1_values =  [score['f1'] for score in test_scores]
    f2_values =  [score['f2'] for score in test_scores]
    scores.append(accuracy_values)
    scores.append(precision_values)
    scores.append(recall_values)
    scores.append(roc_auc_values)
    scores.append(f1_values)
    scores.append(f2_values)
    return scores

def sensitivity_plot(hyperparam, scores, data_name, file_name, hyperparam_string):
    accuracy_values = scores[0]
    precision_values = scores[1]
    recall_values = scores[2]
    roc_auc_values = scores[3]
    f1_values = scores[4]
    f2_values = scores[5]

    if not os.path.exists(str(Config.PLOTS_DIR) + "/" + str(data_name) + "/" + str(file_name)):
        os.makedirs(os.path.join(Config.PLOTS_DIR, data_name, file_name))

    plt.plot(hyperparam, accuracy_values, label='Accuracy')
    plt.plot(hyperparam, precision_values, label='Precision')
    plt.plot(hyperparam, recall_values, label='Recall')
    plt.plot(hyperparam, roc_auc_values, label='ROC AUC Score')
    plt.plot(hyperparam, f1_values, label='F1 Score')
    plt.plot(hyperparam, f2_values, label='F2 Score')
    plt.xlabel(hyperparam_string)
    plt.ylabel('Score')
    plt.legend()
    plt.title(file_name)
    plt.savefig(os.path.join(Config.PLOTS_DIR, data_name, file_name, f"{hyperparam_string}_rf_sensitivity.png"))
    plt.show()