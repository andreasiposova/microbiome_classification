import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from data_loading import load_metadata, load_young_old_labels


def drop_inf_vals(X, y):
    for col in X.columns:
        # find max value of column
        max_value_train = np.nanmax(X[col][X[col] != np.inf])
        # replace inf and -inf in column with max value of column
        X[col].replace([np.inf, -np.inf], 10000, inplace=True)
        # drop the inf values from the test set
        #X = X.replace([np.inf, -np.inf], np.nan).dropna()
    # get the respective y when we drop observations from the test set
    y = y[y.index.isin(X.index)]
    return X, y


def preprocess_data(dataset, metadata_path):#file_name, metadata_path):
    #dataset = globals()[file_name]
    #print(file_name)
    #dataset = file_name
    #subset only young, or old based on which you want to work with, using the labels
    metadata = load_metadata(metadata_path)
    data = dataset.join(metadata)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    X_train, y_train = drop_inf_vals(X_train, y_train)
    X_test, y_test = drop_inf_vals(X_test, y_test)

    #labelEncoder
    #le = preprocessing.LabelEncoder()
    #le.fit(['healthy', 'CRC'])
    #le.classes = np.array(['healthy','CRC'])
    #y_train = le.transform(y_train)
    #y_test = le.transform(y_test)

    label_mapping = {'healthy': 0, 'CRC': 1}
    y_train = [label_mapping[label] for label in y_train]
    y_test = [label_mapping[label] for label in y_test]

    return X_train, X_test, y_train, y_test

def preprocess_huadong(dataset, metadata_path):#file_name, metadata_path):
    #dataset = globals()[file_name]
    #print(file_name)
    #dataset = file_name
    metadata = load_metadata(metadata_path)
    data = dataset.join(metadata)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X, y = drop_inf_vals(X, y)
    X = X.fillna(0)

    label_mapping = {'healthy': 0, 'CRC': 1}
    y = [label_mapping[label] for label in y]


    return X, y




def preprocess_with_y_o_labels(dataset, metadata_path, y_o_labels_filepath, group):
    labels = load_young_old_labels(y_o_labels_filepath)
    meta = load_metadata(metadata_path)
    metadata = labels.join(meta)
    data = dataset.join(metadata)
    if group == 'old':
        data = data.loc[(data['Lon'] == 'oControl') | (data['Lon'] == 'Old') | (data['Lon'] == 'oCRC') | (data['Lon'] == 'oCTRL')]
    if group == 'young:':
        data = data.loc[
            (data['Lon'] == 'yControl') | (data['Lon'] == 'Young') | (data['Lon'] == 'yCRC') | (data['Lon'] == 'yCTRL')]


    data.drop(['Lon'], axis=1, inplace=True)

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    X_train, y_train = drop_inf_vals(X_train, y_train)
    X_test, y_test = drop_inf_vals(X_test, y_test)

    label_mapping = {'healthy': 0, 'CRC': 1}
    y_train = [label_mapping[label] for label in y_train]
    y_test = [label_mapping[label] for label in y_test]

    return X_train, X_test, y_train, y_test

def preprocess_huadong_with_y_o_labels(dataset, metadata_path, y_o_labels_filepath, group):
    labels = load_young_old_labels(y_o_labels_filepath)
    meta = load_metadata(metadata_path)
    metadata = labels.join(meta)
    data = dataset.join(metadata)
    if group == 'old':
        data = data.loc[(data['Lon'] == 'oControl') | (data['Lon'] == 'Old') | (data['Lon'] == 'oCRC') | (data['Lon'] == 'oCTRL')]
    if group == 'young:':
        data.loc[(data['Lon'] == 'yControl') & (data['Lon'] == 'Young')]

    data.drop(['Lon'], axis=1, inplace=True)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X, y = drop_inf_vals(X, y)
    X = X.fillna(0)

    label_mapping = {'healthy': 0, 'CRC': 1}
    y = [label_mapping[label] for label in y]

    return X, y


