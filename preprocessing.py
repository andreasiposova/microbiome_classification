import numpy as np
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from data_loading import load_metadata

def drop_inf_vals(X, y):
    for col in X.columns:
        # find max value of column
        max_value_train = np.nanmax(X[col][X[col] != np.inf])
        # replace inf and -inf in column with max value of column
        X[col].replace([np.inf, -np.inf], max_value_train, inplace=True)
        # drop the inf values from the test set
        X = X.replace([np.inf, -np.inf], np.nan).dropna()
    # get the respective y when we drop observations from the test set
    y = y[y.index.isin(X.index)]
    return X, y


def preprocess_data(dataset, metadata_path):#file_name, metadata_path):
    #dataset = globals()[file_name]
    #print(file_name)
    #dataset = file_name
    metadata = load_metadata(metadata_path)
    data = dataset.join(metadata)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    X_train, y_train = drop_inf_vals(X_train, y_train)
    X_test, y_test = drop_inf_vals(X_test, y_test)

    #labelEncoder
    le = preprocessing.LabelEncoder()
    le.fit(y_train)
    y_train = le.transform(y_train)
    y_test = le.transform(y_test)

    return X_train, X_test, y_train, y_test