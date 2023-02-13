import pandas as pd
import os
from utils import Config

def load_results_table(clf_name):
    results = pd.read_csv(os.path.join(Config.LOG_DIR, f"fudan/final_results/{clf_name}_final_results.csv"))
    return results

rf = load_results_table('rf')
knn = load_results_table('knn')
svm = load_results_table('svm')
xgb = load_results_table('xgb')
all_results = pd.concat(rf, xgb, svm, knn)

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