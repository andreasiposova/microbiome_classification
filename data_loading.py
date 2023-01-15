import pandas as pd
import os

def load_tsv_files(folder):
    # Initialize an empty dictionary to store the variables
    var_dict = {}

    # Iterate over the files in the folder
    for file in os.listdir(folder):
        # Only consider files with the '.tsv' extension
        if file.endswith('.tsv'):
            # Load the data into a pandas DataFrame
            df = pd.read_csv(os.path.join(folder, file), sep='\t')  # skiprows=1
            if len(df.columns) == 1:
                df = pd.read_csv(os.path.join(folder, file), sep='\t', skiprows=1)
                df = df.transpose()
                df.columns = df.iloc[0]
                df.drop(index=df.index[0], axis=0, inplace=True)
                # print(file)
            else:
                # print(file)
                columnNames = df.columns.tolist()
                firstColName = columnNames[0]
                df.set_index(firstColName, inplace=True)
            # Get the file name without the '.tsv' extension
            name = os.path.splitext(file)[0]

            # Assign the DataFrame to a variable with the file name
            globals()[name] = df

            # Add the variable to the dictionary with the file name as the key
            var_dict[name] = globals()[name]

    # Return the dictionary of variables
    return var_dict

def load_data(filepath):
    data_files = load_tsv_files(filepath)

    for key, value in data_files.items():
        globals()[key] = value


def load_metadata(filepath):
    data = pd.read_csv(filepath, sep=",", usecols=["Run", "disease_stat"])
    data.set_index('Run', inplace=True)
    return data