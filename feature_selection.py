from sklearn.datasets import load_iris
from sklearn.model_selection import cross_val_score, KFold
from sklearn.ensemble import RandomForestClassifier

def calculate_feature_importance(X, y, group):
    # initialize the random forest model
    clf = RandomForestClassifier(random_state=1234, n_jobs=-1)

    # initialize the k-fold cross-validation
    kf = KFold(n_splits=3)

    # initialize the feature importance dictionary
    feature_importance = {}

    # iterate over each feature
    # for i in range(X.shape[1]):
    # create a new dataset with one feature removed
    # X_new = X[:, [j for j in range(X.shape[1]) if j != i]]
    for feature in X.columns:
        # create a new dataset by excluding the current feature
        X_new = X[[col for col in X.columns if col != feature]]
        # add the new dataset to the list of new datasets
        # perform k-fold cross validation
        scores = cross_val_score(clf, X_new, y, cv=kf, n_jobs=-1)
        # calculate the mean decrease in accuracy
        mean_decrease_accuracy = 1 - scores.mean()
        # add the feature and its importance to the dictionary
        feature_importance.update({feature: mean_decrease_accuracy})

    # sort the features by importance
    sorted_features = sorted(feature_importance.items(), key=lambda x: x[1], reverse=True)

    # print the sorted features
    for feature, importance in sorted_features:
        print(f"{feature}: {importance:.4f}")

    # print the top 40 most important features
    if group == "old":
        top_features = sorted_features[:40]
    if group == "young":
        top_features = sorted_features[:60]

    # Return the top 40 most important features
    return top_features


def select_features_from_paper(df, group, key):
    if group == "old":
        if key == "genus_relative" or key == "family_relative":
            strings =["g__Peptostreptococcus","g__Parvimonas", "g__Lachnospira", "g__Gemella", "g__Solobacterium", "g__Porphyromonas","g__Monoglobus",
                      "g__Lachnoclostridium","g__Romboutsia","g__Butyricicoccus", "g__Eisenbergiella", "g__Fusicatenibacter", "g__Agathobacter",
                      "g__Limosilactobacillus","g__Roseburia","g__Faecalibacterium", "g__Megamonas", "g__Ruminococcus", "g__Haemophilus",
                      "g__Mogibacterium", "g__Hungatella", "g__Streptococcus", "g__Clostridium sensu stricto", "g__Atopobium", "g__[Eubacterium] ruminantium group",
                      "g_Dorea", "g__Granulicatella", "g__Phocea"]
                      #"f__Family XI", "f__Gemellaceae", "f__Butyricicoccaceae", "f__Porphyromonadaceae", "f__[Eubacterlum] coprostanoligenes group",
                      #"f_[Eubacterium] coprostanoligenes group","f__Aerococcaceae", "f__Selenomonadaceae", "f__Marinifilaceae",
                      #"f__Sutterellaceae", "f__Eggerthellaceae", "f__Monoglobaceae"]
            if key == "family_relative":
                strings = ["f__Family XI", "f__Gemellaceae", "f__Butyricicoccaceae", "f__Porphyromonadaceae", "f__[Eubacterlum] coprostanoligenes group",
                           "f_[Eubacterium] coprostanoligenes group","f__Aerococcaceae",
                           "f__Selenomonadaceae", "f__Marinifilaceae",  "f__Sutterellaceae", "f__Eggerthellaceae", "f__Monoglobaceae"]
    if group=="young":
        if key == "genus_relative":
            strings=["g__Peptostreptococcus","g__Parvimonas", "g__Streptococcus", "g__UCG 002", "g__Lachnospiraceae NC2004 group", "g__Escherichia_Shigella",
                     "g__Stenotrophomonas", "g__Ligilactobacillus", "g__Anaerostipes",
                     "g__Porphyromonas", "g__[Eubacterium] axidoreducens group", "g__Dubosiella", "g__Lachnospiraceae UCG 010",
                     "g__UCG 005", "g__Blautia", "g__Fusobacterium", "q__Phocea", "9__[Eubacterium] ventriosum group",
                     "g__Erysipelotrichaceae CAG 56", "g__Lachnospiraceae ND3007 group", "g__Clastridium sensu stricto",
                     "g__Lactobacillaceae", "g__Agathobacter", "g__[Ruminococcus] gnavus group", "g__Burkholderia Caballeronia Paraburkholderia",
                     "g__Romboutsia", "g__Bifidobacterium", "g__Enterococcus","g__Lawsonella",
                     "g__Incertae_Sedis","g__TM7x", "g__Lachnospira","g__Raoultella","g__Bacteroides",
                     "g__Fusicatenibacter","g__Haemophilus","g__Monoglobus"]

        if key == "family_relative":
                 strings = ["f__[Eubacterium] coprostanoligenes group","g__Atopobium","g__Catenibacterium","f__Butyricicoccaceae", "f__Xanthomonadaceae", "f__Porphyromonadaceae", "f__Oscillospiraceae"
                 "f__Family XI", "f__Lachnospiraceae", "f__Streptococcaceae",  "f__Marinifilaceae", "f__Erysipelatociostridiaceae", "f__Micrococcaceas", "f__Ruminococcaceae",
                 "f__Christensenellaceae", "f__Clostridiaceae","f__Burkholderiaceae ", "f__Staphylococcaceae", "f__Coriobacterlaceae", "f__Caulobacteraceae",
                 "f__[Clostridium] methylpentosum group", "f__Helicobacteracea"]

    elif group=="all":
        strings = ["g__Peptostreptococcus","g__Parvimonas", "g__Streptococcus", "g__UCG 002", "g__Lachnospiraceae NC2004 group", "g__Escherichia_Shigella",
                 "f__Xanthomonadaceae", "f__Porphyromonadaceae", "f__Oscillospiraceae", "g__Stenotrophomonas", "g__Ligilactobacillus", "g__Anaerostipes",
                 "f__Family XI", "g__Porphyromonas", "g__[Eubacterium] axidoreducens group", "g__Dubosiella", "f__Lachnospiraceae", "g__Lachnospiraceae UCG 010",
                 "f__Streptococcaceae", "g__UCG 005", "g__Blautia", "g__Fusobacterium", "q__Phocea", "9__[Eubacterium] ventriosum group",
                 "g__Erysipelotrichaceae CAG 56", "g__Lachnospiraceae ND3007 group", "f__Marinifilaceae", "g__Clastridium sensu stricto", "f__Erysipelatociostridiaceae",
                 "g__Lactobacillaceae", "g__Agathobacter", "g__[Ruminococcus] gnavus group", "f__Micrococcaceas", "g__Burkholderia Caballeronia Paraburkholderia",
                 "g__Romboutsia","f__Christensenellaceae","f__Ruminococcaceae","g__Bifidobacterium","f__Clostridiaceae","g__Enterococcus","g__Lawsonella",
                 "f__Burkholderiaceae ","f__Staphylococcaceae","g__Incertae_Sedis","g__TM7x","f__Coriobacterlaceae","g__Lachnospira","g__Raoultella","g__Bacteroides",
                 "g__Fusicatenibacter","f__Caulobacteraceae","f__[Clostridium] methylpentosum group","f__Helicobacteracea","g__Haemophilus","g__Monoglobus",
                 "f__[Eubacterium] coprostanoligenes group","g__Atopobium","g__Catenibacterium","g__Gemella", "g__Solobacterium",
                 "g__Porphyromonas", "g__Lachnoclostridium", "g__Romboutsia", "g__Butyricicoccus", "g__Eisenbergiella",
                 "g__Limosilactobacillus", "g__Roseburia", "g__Faecalibacterium", "g__Megamonas", "g__Ruminococcus",
                 "g__Mogibacterium", "g__Hungatella", "g__Clostridium sensu stricto", "g__[Eubacterium] ruminantium group",
                 "g_Dorea", "g__Granulicatella", "f__Gemellaceae", "f__Butyricicoccaceae", "f__[Eubacterlum] coprostanoligenes group",
                 "f_[Eubacterium] coprostanoligenes group", "f__Aerococcaceae", "f__Selenomonadaceae", "f__Marinifilaceae", "f__Sutterellaceae", "f__Eggerthellaceae", "f__Monoglobaceae"]

    cols = df.columns[df.columns.str.contains('|'.join(strings))].tolist()

    # Select the columns that contain any of the 40 strings in the list
    df_filtered = df[cols]

    return df_filtered

#This will return a new DataFrame df_filtered that contains only the columns where the column name contains any of the 40 strings in the list.


    """
    clf = RandomForestClassifier(random_state=42)

    # fit the model on the training data
    clf.fit(X, y)

    # get the feature importances
    feature_importance = clf.feature_importances_

    # get the feature names
    feature_names = X.columns

    # create a dictionary of feature importances
    feature_importance_dict = {feature_names[i]: feature_importance[i] for i in range(len(feature_names))}

    # sort the features by importance
    sorted_features = sorted(feature_importance_dict.items(), key=lambda x: x[1], reverse=True)
    top_features = sorted_features[:40]

    return top_features


    """