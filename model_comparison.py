import pandas as pd
import os
from utils import Config
import matplotlib
matplotlib.use('Agg')
from matplotlib import pyplot as plt

def load_results_table(clf_name):
    results = pd.read_csv(os.path.join(Config.LOG_DIR, f"fudan/final_results/{clf_name}_final_results.csv"))
    return results

rf = load_results_table('rf')
knn = load_results_table('knn')
svm = load_results_table('svm')
xgb = load_results_table('xgb')
all_results = pd.concat([rf, svm, knn, xgb])
curr_index = all_results.columns.get_loc("roc_auc")

# Insert the column at a new position
new_position = 5
all_results.insert(new_position, "roc_auc", all_results.pop("roc_auc"))
all_results.drop('Unnamed: 0', inplace=True, axis=1)
all_results.to_csv(os.path.join(Config.LOG_DIR, 'fudan/final_results/final_results.csv'))

groups = ['old', 'young', 'all']
features = ['selected_features', 'all_features']

def compare_models_selected_features(all_results):
    groups = ['old', 'young', 'all']
    for i in groups:

        df_filtered = all_results[(all_results['samples'] == i)]
        df_filtered = df_filtered[(df_filtered['Feature Abundance Limits'] == 'no transformation')]
        df_filtered = df_filtered[(df_filtered['features'] == 'selected')]
                                  #& (all_results['features'] == 'selected')]


        # Get the values of column 'roc_auc' from the filtered dataframe
        roc_auc_values = list(df_filtered['roc_auc'].values)
        classifiers = list(df_filtered['classifier'])

        # Create the barplot
        plt.bar(x = classifiers, height = roc_auc_values)
        plt.xlabel('')
        plt.ylabel('roc_auc')
        plt.ylim((0,1))
        plt.title('ROC AUC')

        plt.savefig(os.path.join(Config.PLOTS_DIR, f'fudan/final_results/{i}/selected_features/barplot_roc_auc.png'))
        plt.close()

#compare_models_selected_features(all_results)

def compare_models_all_features(all_results):
    groups = ['old', 'young', 'all']
    for i in groups:

        df_filtered = all_results[(all_results['samples'] == i)]
        df_filtered = df_filtered[(df_filtered['Feature Abundance Limits'] == 'no transformation')]
        df_filtered = df_filtered[(df_filtered['features'] == 'all')]
                                  #& (all_results['features'] == 'selected')]


        # Get the values of column 'roc_auc' from the filtered dataframe
        roc_auc_values = list(df_filtered['roc_auc'].values)
        classifiers = list(df_filtered['classifier'])

        # Create the barplot
        plt.bar(x = classifiers, height = roc_auc_values)
        plt.xlabel('')
        plt.ylabel('roc_auc')
        plt.ylim((0,1))
        plt.title('ROC AUC')

        plt.savefig(os.path.join(Config.PLOTS_DIR, f'fudan/final_results/{i}/all_features/barplot_roc_auc.png'))
        plt.close()


def compare_models_selected(all_results):
    groups = ['old', 'young', 'all']
    for i in groups:

        df_filtered = all_results[(all_results['samples'] == i)]
        df_filtered = df_filtered[(df_filtered['Feature Abundance Limits'] == 'no transformation')]
        df_filtered = df_filtered[(df_filtered['features'] == 'selected')]
        # & (all_results['features'] == 'selected')]


        # Get the values of column 'roc_auc' from the filtered dataframe
        df_filtered = df_filtered[['classifier', 'roc_auc', 'accuracy', 'precision', 'recall', 'f1', 'f2']]

        cols = df_filtered.columns
        cols = cols[1:]
        for col in cols:
            df_filtered[col] = (df_filtered[col] * 100).round(2)

        df_filtered.to_csv(os.path.join(Config.LOG_DIR, f"fudan/final_results/{i}_selected_features_model_comparison.csv"))


compare_models_selected(all_results)























