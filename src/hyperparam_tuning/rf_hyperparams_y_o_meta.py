import pandas as pd
from src.utils.data_loading import load_metadata
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split


yang_metadata = load_metadata("../../data/Yang_PRJNA763023/metadata.csv")
#yang_data = load_files("GASTRIC/Yang_PRJNA763023/Yang_PRJNA763023_PE_1")
#load_yang()
metaanalysis_metadata = pd.read_csv("../../data/CRC_metaanalysis/metadata.tsv", sep="\t")
metaanalysis_data = pd.read_csv("../../data/CRC_metaanalysis/features.tsv", sep="\t")
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



#le = preprocessing.LabelEncoder()
#le.fit(y)
#y = le.transform(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=1234)
label_mapping = {'healthy': 0, 'CRC': 1}
y_train = y_train.iloc[:,0]
y_test = y_test.iloc[:,0]
y_train = [label_mapping[label] for label in y_train]
y_test = [label_mapping[label] for label in y_test]

clf = RandomForestClassifier(n_estimators= 180, max_depth=20, min_samples_leaf=20, random_state=1234) #class_weight={0: 1}
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)
acc = accuracy_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred)
print("hi")