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
