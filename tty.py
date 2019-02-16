import sklearn
import sklearn.datasets
import sklearn.ensemble
import numpy as np
import pandas as pd
import lime
import lime.lime_tabular
import csv
from IPython.core.display import display, HTML

np.random.seed(1)


def run():#import dataset
    data = np.genfromtxt('G:\StudentsPerformance.csv', delimiter=',', dtype=str)
    #create labels
    labels = data[:,3]
    le= sklearn.preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    class_names = le.classes_
    data=np.concatenate((data[:,0:3],data[:,4:]),axis=1)
    #print(data)
    categorical_features = [0,1,2,3,4,5]

    feature_names = ["Gender", "Race",  "Parent Education", "Lunch", "Test preparation","Math", "Reading", "Writing"]

    #create labels' names
    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_

    data = data.astype(float)
    encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)
    np.random.seed(1)
    train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)
    encoder.fit(data)
    encoded_train = encoder.transform(train)

    import xgboost
    gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
    gbtree.fit(encoded_train, labels_train)

    sklearn.metrics.accuracy_score(labels_test, gbtree.predict(encoder.transform(test)))
    predict_fn = lambda x: gbtree.predict_proba(encoder.transform(x)).astype(float)

    explainer = lime.lime_tabular.LimeTabularExplainer(train ,feature_names = feature_names,class_names=class_names,
                                                       categorical_features=categorical_features,
                                                       categorical_names=categorical_names, kernel_width=3)

    np.random.seed(1)
    i = 199
    exp = explainer.explain_instance(test[i], predict_fn, num_features=5)
    exp.save_to_file("static/limeresult.html",show_all=False)
    return 0