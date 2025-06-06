import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from src.data.loader import load_metadata, load_young_old_labels


def replace_inf_vals(X, y):
    for col in X.columns:
        # find max value of column
        max_value_train = np.nanmax(X[col][X[col] != np.inf])
        min_value_train = np.nanmin(X[col][X[col] != np.inf])
        # replace inf and -inf in column with max value of column
        X[col] = X[col].replace([np.inf, -np.inf], max_value_train)
        X[col] = X[col].replace([-np.inf], min_value_train)
    return X, y


def preprocess_data(dataset, metadata_path):
    #subset only young, or old based on which you want to work with, using the labels
    metadata = load_metadata(metadata_path)
    data = dataset.join(metadata)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X_train = X
    y_train = y
    X_test = pd.DataFrame()
    y_test = pd.DataFrame()
    X_train, y_train = replace_inf_vals(X_train, y_train)

    label_mapping = {'healthy': 0, 'CRC': 1}
    y_train = [label_mapping[label] for label in y_train]

    return X_train, X_test, y_train, y_test

def preprocess_huadong(dataset, metadata_path):
    metadata = load_metadata(metadata_path)
    data = dataset.join(metadata)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    X, y = replace_inf_vals(X, y)
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
        data1 = data.loc[(data['Lon'] == 'oControl')]
        data2 = data.loc[(data['Lon'] == 'Old')]
        data3 = data.loc[(data['Lon'] == 'oCRC')]
        data4 = data.loc[(data['Lon'] == 'oCTRL')]
        dataset = pd.concat([data1, data2, data3, data4])
    if group == 'young':
        data1 = data.loc[(data['Lon'] == 'yControl')]
        data2 = data.loc[(data['Lon'] == 'Young')]
        data3 = data.loc[(data['Lon'] == 'yCRC')]
        data4 = data.loc[(data['Lon'] == 'yCTRL')]
        dataset = pd.concat([data1, data2, data3, data4])


    dataset.drop(['Lon'], axis=1, inplace=True)

    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    X_train = X
    y_train = y
    X_test = pd.DataFrame()
    y_test = pd.DataFrame()

    label_mapping = {'healthy': 0, 'CRC': 1}
    y_train = [label_mapping[label] for label in y_train]

    return X_train, X_test, y_train, y_test

def preprocess_huadong_with_y_o_labels(dataset, metadata_path, y_o_labels_filepath, group):
    labels = load_young_old_labels(y_o_labels_filepath)
    meta = load_metadata(metadata_path)
    metadata = labels.join(meta)
    data = dataset.join(metadata)
    if group == 'old':
        data1 = data.loc[(data['Lon'] == 'oControl')]
        data2 = data.loc[(data['Lon'] == 'Old')]
        data3 = data.loc[(data['Lon'] == 'oCRC')]
        data4 = data.loc[(data['Lon'] == 'oCTRL')]
        dataset = pd.concat([data1, data2, data3, data4])
    if group == 'young':
        data1 = data.loc[(data['Lon'] == 'yControl')]
        data2 = data.loc[(data['Lon'] == 'Young')]
        data3 = data.loc[(data['Lon'] == 'yCRC')]
        data4 = data.loc[(data['Lon'] == 'yCTRL')]
        dataset = pd.concat([data1, data2, data3, data4])

    dataset.drop(['Lon'], axis=1, inplace=True)
    X = dataset.iloc[:, :-1]
    y = dataset.iloc[:, -1]

    X, y = replace_inf_vals(X, y)
    X = X.fillna(0)

    label_mapping = {'healthy': 0, 'CRC': 1}
    y = [label_mapping[label] for label in y]

    return X, y


def full_preprocessing_y_o_labels(data, huadong_data1, huadong_data2, key, yang_metadata_path, young_old_labels_path, group):
    X_train_1, X_test_1, y_train, y_test = preprocess_with_y_o_labels(data[key], yang_metadata_path,young_old_labels_path, group)
    X_train, y_train = replace_inf_vals(X_train_1, y_train)
    X_test, y_test = replace_inf_vals(X_test_1, y_test)
    X_h1, y_h1 = preprocess_huadong_with_y_o_labels(huadong_data1[key], yang_metadata_path, young_old_labels_path,group)
    X_h2, y_h2 = preprocess_huadong_with_y_o_labels(huadong_data2[key], yang_metadata_path, young_old_labels_path,group)

    X_val_1 = pd.concat([X_h1, X_h2])
    y_val = y_h1 + y_h2
    X_val_1, y_val = replace_inf_vals(X_val_1, y_val)

    return X_train_1, X_test_1, X_val_1, y_train, y_test, y_val

def remove_correlated_features(df, threshold):
    corr_matrix = df.corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(np.bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    df = df.drop(df[to_drop], axis=1)
    return df



def apply_feature_abundance_limits(main_df, type):
    if type == 'low':
        cla = [-np.inf, 0.00001, 0.0001, 0.001, 0.01, 0.1, np.inf]
    elif type == 'medium':
        cla = [-np.inf, 0.00003, 0.0003, 0.003, 0.03, 0.3, np.inf]
    elif type == 'high':
        cla = [-np.inf, 0.00005, 0.0005, 0.005, 0.05, 0.5, np.inf]
    main_df = pd.DataFrame(main_df)
    main_index = list(main_df.index.values)
    main_columns = list(main_df.columns.values)
    main_df = pd.DataFrame(np.digitize(main_df, cla), index=list(main_index), columns=list(main_columns)).subtract(1)
    return main_df