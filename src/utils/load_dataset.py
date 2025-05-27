import os
import pandas as pd
from scipy.io import arff
from sklearn.preprocessing import LabelEncoder
from .config import Config


def load_training_dataset(return_label_encoder=False):
    #raw_data = arff.loadarff(os.path.join(Config.DATA_DIR, "dataset_37_diabetes.arff"))

    df = pd.DataFrame(raw_data[0])
    target = df["class"]
    X = df.drop("class", axis=1)
    le = LabelEncoder()
    y = le.fit_transform(target)
    if return_label_encoder:
        return X, y, le
    else:
        return X, y
