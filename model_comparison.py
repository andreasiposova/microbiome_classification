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

def compare_models(all_results):
    for i in groups:

        df_filtered = all_results[(all_results['group'] == i) & (all_results['Feature Abundance Limits'] == 'no_transformation')
                                  & (all_results['features'] == 'selected')]


    # Get the values of column 'roc_auc' from the filtered dataframe
    roc_auc_values = df_filtered['roc_auc'].values

    # Create the barplot
    plt.bar(['old - no_transformation'], [roc_auc_values.mean()], yerr=[roc_auc_values.std()])
    plt.xlabel('group and fal')
    plt.ylabel('roc_auc')
    plt.title('ROC AUC Barplot')
    plt.show()