import numpy as np
import pandas as pd
import os
import sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_validate, RepeatedStratifiedKFold, train_test_split


from sklearn import preprocessing
from sklearn.metrics import precision_recall_fscore_support
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_auc_score
from sklearn.metrics import f1_score

from preprocess_data import preprocess
from load_data import load_data, load_files, load_yang

yang_metadata = load_data("GASTRIC/Yang_PRJNA763023/metadata.csv")
#yang_data = load_files("GASTRIC/Yang_PRJNA763023/Yang_PRJNA763023_PE_1")
#load_yang()
metaanalysis_metadata = pd.read_csv("GASTRIC/CRC_metaanalysis/metadata.tsv", sep="\t")
metaanalysis_data = pd.read_csv("GASTRIC/CRC_metaanalysis/features.tsv", sep="\t")
metaanalysis_data = metaanalysis_data.transpose()
metaanalysis_data.columns = metaanalysis_data.iloc[0]
metaanalysis_data = metaanalysis_data.drop(metaanalysis_data.index[0])

metaanalysis_metadata = metaanalysis_metadata.loc[metaanalysis_metadata['study'] == "Yang"]
uniqueSamples = metaanalysis_metadata['Run'].nunique()
metaanalysis_data.index.name = 'Run'
metaanalysis_data.reset_index(inplace=True)
data = metaanalysis_metadata.merge(metaanalysis_data, how='inner', on='Run')
data = data.drop(columns=['Unnamed: 0', 'age', 'gender', 'cancer_stage', 'bmi', 'diagnosis', 'race'])
X = data.iloc[:, 5:]
y = data.iloc[:, 4:5]
le = preprocessing.LabelEncoder()
le.fit(y)
y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
clf = RandomForestClassifier(n_estimators= 100, max_depth=50, random_state=1234, class_weight={0: 1})
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
nu = np.unique(y_test, return_counts=True)

p_r_f1_support = precision_recall_fscore_support(y_test, y_pred, average='macro')
acc = accuracy_score(y_test, y_pred)
cm = confusion_matrix(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
print(uniqueSamples)


"""path = "GASTRIC/Yang_PRJNA763023/Yang_PRJNA763023_PE_1/parsed/normalized_results/"
files = os.listdir(path)
for file in files:
    if file == "class_relative.tsv":
        class_relative = pd.read_csv(path + file, sep="\t", skiprows=1)
        class_relative = class_relative.transpose()
    elif file == "domain_relative.tsv":
        domain_relative = pd.read_csv(path + file, sep="\t", skiprows=1)
        domain_relative = domain_relative.transpose()
    elif file == "order_relative.tsv":
        order_relative = pd.read_csv(path + file, sep="\t", skiprows=1)
        order_relative = order_relative.transpose()"""

#preprocess()

#print (peters_data)

#assert my_data["hello"] == "goodbye"

    #foo = os.path.splitext(file)[0]
    #exec(foo + data)
    #all_data.append(data)
#print(data)

