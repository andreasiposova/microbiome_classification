import argparse
import os
import pandas as pd
import numpy as np


from utils import setup_logging, Config
from preprocessing import preprocess_data

from datetime import datetime

from visualization import sensitivity_plot, get_scores, heatmap_plot

from data_loading import load_tsv_files

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, \
    fbeta_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate
from sklearn.base import clone
from sklearn.pipeline import Pipeline

import matplotlib.pyplot as plt

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

def sensitivity_analysis(filepath, data_name):
    full_results = []
    data = load_tsv_files(fudan_filepath)
    for key in data:
        # load your data
        X_train, X_test, y_train, y_test = preprocess_data(data[key], yang_metadata_path)

    # define estimator
        estimator = Pipeline([("model",RandomForestClassifier(random_state=1234, n_estimators=100, max_depth=5, min_samples_split=6, min_samples_leaf=1, max_features=None))])
        # define a range of values for the maximum depth
        param_ranges = {
            "max_depth": [5, 10], #, 20, 30, 40, 50, 60, 70, 80, 90, 100,
            "n_estimators": [5, 10], #np.linspace(100, 3000, 10, dtype=int),
            "min_samples_split": [2, 4], #range(2, 20),
            "min_samples_leaf": [2, 5], #range(1, 6),
            "max_features": ["sqrt"] #, "log2", None]
        }

        # initialize an empty list to store the performance scores
        scores = []

        # iterate over the range of values for maximum depth
        for param in param_ranges:
            print(f"Running sensitivity analysis on parameter {param}")
            accuracy = []
            precision = []
            recall = []
            roc_auc = []
            f1 = []

            for value in param_ranges[param]:
                model = clone(estimator) #clone(estimator)
                model.set_params(**{f"model__{param}": value})
                cv = RepeatedStratifiedKFold(n_splits=10, random_state=1234)
                scores = cross_validate(model, X_train, y_train, scoring=["accuracy", "precision", "recall", "roc_auc", "f1"], cv=cv, n_jobs=-1)
                accuracy.append(scores["test_accuracy"].mean())
                precision.append(scores["test_precision"].mean())
                recall.append(scores["test_recall"].mean())
                roc_auc.append(scores["test_roc_auc"].mean())
                f1.append(scores["test_f1"].mean())
            plt.figure(figsize=(7, 5))
            x_str = [str(value) for value in param_ranges[param]]
            if "None" in x_str or "True" in x_str or "False" in x_str:
                x = x_str
            else:
                x = param_ranges[param]
            plt.plot(x, accuracy, label="Accuracy")
            plt.plot(x, precision, label="Precision")
            plt.plot(x, recall, label="Recall")
            plt.plot(x, roc_auc, label="ROC AUC")
            plt.plot(x, f1, label="F1")
            plt.xlabel(param)
            plt.ylabel("score")
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(os.path.join(Config.PLOTS_DIR, data_name, key, f"{param}_rf_sensitivity.png")))

#sensitivity_analysis(fudan_filepath, FUDAN)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_name', type=str, default=HUADONG1)
    parser.add_argument('--filepath', type=str, default=huadong_filepath_1)

    args = parser.parse_args()
    data_name = args.data_name
    if data_name == FUDAN:
        sensitivity_analysis(data_name=args.data_name, filepath=args.filepath)
    elif data_name == HUADONG1:
        sensitivity_analysis(data_name=args.data_name, filepath=args.filepath)
    else:
        raise ValueError()


