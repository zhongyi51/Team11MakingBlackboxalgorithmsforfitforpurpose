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

def readfromcsv(address):
    data = np.genfromtxt(address, delimiter=',', dtype=str)
    #print(data[0])
    labels = []
    for i in range(0, len(data[0])):
        labels.append(data[0][i].strip('"'))
    datae=data[1:,:]
    for i in range(0,np.size(datae,0)):
        for r in range(0,np.size(datae,1)):
            datae[i][r]=datae[i][r].strip('"')
    #print(data[0])
    return labels,datae

def readfromtlabeltext(text,labels):
    i=labels.index(text)
    return i


def readfromfeaturetext(features, labels):
    dataset=features.split(',')
    i=[]
    for e in dataset:
        i.append(labels.index(e))
    return i

def run(categoricalfeaturestext,labelstext,address,instance):#import dataset
    #read csv
    instance=np.array(instance.split(','))
    feature_names,data=readfromcsv(address)
    labelsindex=readfromtlabeltext(labelstext,feature_names)
    feature_names.remove(labelstext)
    categorical_features=readfromfeaturetext(categoricalfeaturestext,feature_names)
    #del(instance[labelsindex])
    #create labels
    labels = data[:,labelsindex]
    le= sklearn.preprocessing.LabelEncoder()
    le.fit(labels)
    labels = le.transform(labels)
    class_names = le.classes_
    data=np.concatenate((data[:,0:labelsindex],data[:,labelsindex+1:]),axis=1)
    #instance=np.concatenate((instance[0:labelsindex],instance[labelsindex+1:]),axis=0)
    #categorical_features = [0,1,2,3,4,5]
    #feature_names = ["Gender", "Race",  "Parent Education", "Lunch", "Test preparation","Math", "Reading", "Writing"]
    #create labels' names
    data=np.append(data,[instance],axis=0)#add instance to data to transform it
    categorical_names = {}
    for feature in categorical_features:
        le = sklearn.preprocessing.LabelEncoder()
        le.fit(data[:, feature])
        data[:, feature] = le.transform(data[:, feature])
        categorical_names[feature] = le.classes_

    #take instance out
    data=data[0:len(data)-1]
    instance=data[-1]
    #numeral transform
    data = data.astype(float)
    instance=instance.astype(float)
    encoder = sklearn.preprocessing.OneHotEncoder(categorical_features=categorical_features)
    np.random.seed(1)
    train, test, labels_train, labels_test = sklearn.model_selection.train_test_split(data, labels, train_size=0.80)
    encoder.fit(data)
    encoded_train = encoder.transform(train)

    #predicting
    import xgboost
    gbtree = xgboost.XGBClassifier(n_estimators=300, max_depth=5)
    gbtree.fit(encoded_train, labels_train)

    sklearn.metrics.accuracy_score(labels_test, gbtree.predict(encoder.transform(test)))
    predict_fn = lambda x: gbtree.predict_proba(encoder.transform(x)).astype(float)
    explainer = lime.lime_tabular.LimeTabularExplainer(train ,feature_names = feature_names,class_names=class_names,
                                                       categorical_features=categorical_features,
                                                       categorical_names=categorical_names, kernel_width=3)

    np.random.seed(1)
    print(test[1],instance)
    #i = int(instance)
    #print(instance)
    exp = explainer.explain_instance(instance, predict_fn, num_features=5)
    exp.save_to_file("static/limeresult.html",show_all=False)
    return 0