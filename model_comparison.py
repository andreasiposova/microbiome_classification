import pandas as pd
import os
from utils import Config

results = pd.read_csv(os.path.join(Config.LOG_DIR, "final_results/final_results.csv"))

