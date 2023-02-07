import numpy as np
import pandas as pd
import sklearn
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler

from data_loading import load_metadata, load_young_old_labels


def replace_inf_vals(X, y):
    for col in X.columns:
        # find max value of column
        max_value_train = np.nanmax(X[col][X[col] != np.inf])
        min_value_train = np.nanmin(X[col][X[col] != np.inf])
        # replace inf and -inf in column with max value of column
        X[col].replace([np.inf, -np.inf], max_value_train, inplace=True)
        X[col].replace([-np.inf], min_value_train, inplace=True)
        # drop the inf values from the test set
        #X = X.replace([np.inf, -np.inf], np.nan).dropna()
    # get the respective y when we drop observations from the test set
    #y = y[y.index.isin(X.index)]
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

    X_train, y_train = replace_inf_vals(X_train, y_train)
    X_test, y_test =replace_inf_vals(X_test, y_test)

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

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)

    #X_train, y_train = replace_inf_vals(X_train, y_train)
    #X_test, y_test = replace_inf_vals(X_test, y_test)


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
    if group == 'young':
        data = data.loc[(data['Lon'] == 'yControl') | (data['Lon'] == 'Young')| (data['Lon'] == 'yCRC') | (data['Lon'] == 'yCTRL')]

    data.drop(['Lon'], axis=1, inplace=True)
    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

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



def apply_feature_abundance_limits(main_df):
    cla = [-np.inf, 0.00005, 0.0005, 0.005, 0.05, 0.5, np.inf]
    main_df = pd.DataFrame(main_df)
    main_index = list(main_df.index.values)
    main_columns = list(main_df.columns.values)
    main_df = pd.DataFrame(np.digitize(main_df, cla), index=list(main_index), columns=list(main_columns)).subtract(1)
    return main_df


