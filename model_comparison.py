import numpy as np
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

compare_models_selected_features(all_results)

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

compare_models_selected_features(all_results)
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

def fal_barplots(all_results):
    colors = ['#F5CBA7', '#B8DAF0', '#FFC300', '#C8C8A9', '#F3E5AB', '#AED6F1']

    groups = ['old', 'young', 'all']
    features = ['all', 'selected']
    classifiers = ['XGB', 'SVM', 'RF', 'KNN']
    for classifier in classifiers:
        for group in groups:
            for j in features:
                df_filtered = all_results[(all_results['samples'] == group)]
                df_filtered = df_filtered[(df_filtered['features'] == j)]
                df_filtered = df_filtered[(df_filtered['classifier'] == classifier)]
                #df_filtered = df_filtered[(df_filtered['Feature Abundance Limits'] == 'no transformation')]
                levels = list(df_filtered['Feature Abundance Limits'])
                #levels = levels[0:4]
        # & (all_results['features'] == 'selected')]


        # Get the values of column 'roc_auc' from the filtered dataframe
                df_filtered = df_filtered[['roc_auc', 'accuracy', 'precision', 'recall', 'f1', 'f2']]
                accuracy = list(df_filtered['accuracy'])
                precision = list(df_filtered['precision'])
                recall = list(df_filtered['recall'])
                f1 = list(df_filtered['f1'])
                f2 = list(df_filtered['f2'])
                roc_auc = list(df_filtered['roc_auc'])
                bar_width = 0.08
                bar_spacing = 0.03

                x_pos = np.arange(len(levels))

                # Create grouped barplot

                fig, ax = plt.subplots(figsize=(10,8))
                ax.bar(x_pos - 3 * bar_width - 3 * bar_spacing, roc_auc, width=bar_width, label='ROC AUC', color=colors[5])
                ax.bar(x_pos - 2 * bar_width - 2 * bar_spacing, accuracy, width=bar_width, label='Accuracy', color=colors[0])
                ax.bar(x_pos - 1 * bar_width - 1 * bar_spacing, precision, width=bar_width, label='Precision',color=colors[1])
                ax.bar(x_pos + 0 * bar_width , recall, width=bar_width, label='Recall', color=colors[2])
                ax.bar(x_pos + 1 * bar_width + 2 * bar_spacing, f1, width=bar_width, label='F1', color=colors[3])
                ax.bar(x_pos + 2 * bar_width + 3 * bar_spacing, f2, width=bar_width, label='F2', color=colors[4])


                # Add legend and labels
                ax.set_ylabel('Performance')
                ax.set_xlabel('Levels')
                ax.set_xticks(x_pos)
                ax.set_xticklabels(levels)

                #ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), ncol=6, frameon=True, framealpha=0.8, fancybox=True, fontsize='small', borderaxespad=0.5)
                ax.legend(loc='upper center', bbox_to_anchor=(0.5, 0.99), ncol=6, frameon=True, framealpha=0.8,
                                   fancybox=True, fontsize='small', borderaxespad=0.25)

                # Set the legend background color and border properties
                #legend.get_frame().set_facecolor('white')
                #legend.get_frame().set_linewidth(0.5)
                #legend.get_frame().set_edgecolor('black')


                plt.ylim((0,1))
                plt.title(f"Impact of Feature Abundance Limits on the {classifier} performance")
                plt.savefig(os.path.join(Config.PLOTS_DIR, 'fudan/final_results', group, f"{j}_features", classifier,
                                         f"{classifier}_fal_comparison.png"))
                plt.close(fig)




fal_barplots(all_results)




def fal_FN_plots(all_results):
    groups = ['old', 'young', 'all']
    features = ['all', 'selected']
    classifiers = ['XGB', 'SVM', 'RF', 'KNN']
    for classifier in classifiers:
        for group in groups:
            for j in features:
                df_filtered = all_results[(all_results['samples'] == group)]
                df_filtered = df_filtered[(df_filtered['features'] == j)]
                df_filtered = df_filtered[(df_filtered['classifier'] == classifier)]
                #df_filtered = df_filtered[(df_filtered['Feature Abundance Limits'] == 'no transformation')]
                levels = list(df_filtered['Feature Abundance Limits'])
                fns = list(df_filtered['FN'])


                # Create the barplot
                plt.bar(x = levels, height = fns)
                plt.xlabel('')
                plt.ylabel('Number of FN predictions')
                plt.title('Impact of Feature Abundance Limits on FN predictions')

                plt.savefig(os.path.join(Config.PLOTS_DIR, 'fudan/final_results', group, f"{j}_features", classifier,
                                         f"{classifier}_fal_FNs.png"))
                plt.close()



fal_FN_plots(all_results)


def final_results_table_100(all_results):

    cols = ['roc_auc', 'accuracy', 'precision', 'recall', 'f1', 'f2', 'CI lower', 'CI upper']
    for col in cols:
        all_results[col] = (all_results[col] * 100).round(2)
    tpfptnfn = ['TP', 'TN', 'FP', 'FN']
    for col in tpfptnfn:
        all_results[col] = (all_results[col].astype('int'))

    all_results.to_csv(os.path.join(Config.LOG_DIR, f"fudan/final_results/final_results100.csv"))

final_results_table_100(all_results)













