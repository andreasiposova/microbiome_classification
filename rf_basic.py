import argparse
import os
import pandas as pd
import numpy as np

from utils import setup_logging, Config
from preprocessing import preprocess_data, preprocess_huadong, preprocess_with_y_o_labels, \
    preprocess_huadong_with_y_o_labels

from datetime import datetime
from data_loading import load_tsv_files

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, roc_auc_score, f1_score, \
    fbeta_score
from sklearn.model_selection import GridSearchCV, RepeatedStratifiedKFold, cross_validate
from sklearn.base import clone
from sklearn.pipeline import Pipeline

import matplotlib
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


full_results = []
data = load_tsv_files(fudan_filepath)
huadong_data1 = load_tsv_files(huadong_filepath_1)
huadong_data2 = load_tsv_files(huadong_filepath_2)
for key in data:
    if key == "genus_relative":
        # load your data
        X_train, X_test, y_train, y_test = preprocess_with_y_o_labels(data[key], yang_metadata_path,
                                                                      young_old_labels_path, 'young')
        X_h1, y_h1 = preprocess_huadong_with_y_o_labels(huadong_data1[key], yang_metadata_path, young_old_labels_path,
                                                        'young')
        X_h2, y_h2 = preprocess_huadong_with_y_o_labels(huadong_data2[key], yang_metadata_path, young_old_labels_path,
                                                        'young')
        # = preprocess_huadong(huadong_data1[key], yang_metadata_path)
        # X_h2, y_h2 = preprocess_huadong(huadong_data2[key], yang_metadata_path)
        X_val = pd.concat([X_h1, X_h2])
        y_val = y_h1 + y_h2
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
        X_test
        rf = RandomForestClassifier(n_estimators = 180, min_samples_leaf=20)
        rf.fit(X_train, y_train)
        y_pred_test = rf.predict(X_test)
        y_pred_val = rf.predict(X_val)

        acc_test = accuracy_score(y_test,y_pred_test)
        roc_auc_test = roc_auc_score(y_test,y_pred_test)

        acc_val = accuracy_score(y_val, y_pred_val)
        roc_auc_val = roc_auc_score(y_val, y_pred_val)

        print("hi")