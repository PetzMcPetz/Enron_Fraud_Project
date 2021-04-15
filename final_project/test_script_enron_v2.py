from __future__ import division#, print_function

import poi_id_v2 as poi
import pprint
import pandas as pd
import numpy as np
import seaborn as sns
import sys
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
import tester_project as tester

import matplotlib.pyplot as plt
import pickle

from sklearn.model_selection import train_test_split, GridSearchCV

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score
from sklearn.feature_selection import SelectKBest, chi2, f_classif
#1 Load Data
data_dict = poi.load_dataset()

#2 Check Data
nan_email_list, nan_all_list, poi_data_dict = poi.check_dataset(data_dict)

print len(data_dict)
print len(poi_data_dict.keys())

pprint.pprint(poi_data_dict.keys())
pprint.pprint(nan_all_list)

#3 Plot Data / Find Outlier
feature_list = ["salary", "bonus"]

df=poi.df_features(data_dict,feature_list)

#plt.subplots(figsize=(12, 6))
#plt.scatter(df[feature_list[0]], df[feature_list[0]])
#plt.xlabel(feature_list[0])
#plt.ylabel(feature_list[1])
#plt.show()

### Check for NaN
df=poi.nan_check(data_dict)

### Check Outlier
poi.check_outlier(data_dict, 'salary')

### Remove Outlier
outlier = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
my_dataset = poi.remove_outlier(data_dict, outlier)

########################################################## Modified Data Dict: Create Labels and Features
### Create new Feature
my_dataset = poi.create_new_message_feature(my_dataset)

excluded_features   = ['email_address']
feature_list        = poi.get_feature(my_dataset, excluded_features)

########################################################## Modified Data Dict: Create Labels and Features
data = featureFormat(my_dataset, feature_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

########################################################## Modified Data Dict: Select Kbest features
k=len(features[0])

kBest_features, sorted_scores, kBest_feature_list = poi.get_KBest(features, labels, k, feature_list)

sorted_feature_list = ['poi'] + kBest_feature_list

print pd.DataFrame(sorted_scores,columns=['Feature','KBest Score'])

########################################################## Test different Classifier

clf_dict = {'DecisionTree': DecisionTreeClassifier(class_weight=None, criterion='entropy', max_depth=None,
                            max_features=None, max_leaf_nodes=None,
                            min_impurity_decrease=0.0, min_impurity_split=None,
                            min_samples_leaf=1, min_samples_split=3,
                            min_weight_fraction_leaf=0.0, presort=False, random_state=0,
                            splitter='best'),
            'GaussianNB':   GaussianNB(),
            'SVC':          SVC(C=300, cache_size=200, class_weight='balanced', coef0=0.0,
                            decision_function_shape='ovr', degree=3, gamma=0.001, kernel='rbf',
                            max_iter=-1, probability=False, random_state=None, shrinking=True,
                            tol=0.001, verbose=False)}

k=[4,5,6,7,8,9,10,11,12,13,14,15,16]

clf_key = 'SVC'

f1, ii, selected_features_list, pipe, df_k_result, df_k_best_result = poi.clf_func(k, clf_dict, sorted_feature_list, my_dataset, clf_key)
print ii
print ''
print 'Result Summary for K-Features'
print ''
print df_k_result
print ''
print 'Best Results for ', ii, ' features'
print ''
print 'Selected Fetaures: ',len(selected_features_list[1:]), selected_features_list[1:]
print ''
print df_k_best_result
print ''
########################################################## Export Project Data

# Save project data as pkl files
tester.dump_classifier_and_data(pipe, my_dataset, selected_features_list)
# Load pkl files for evaluation
tester.test_classifier(pipe, my_dataset, selected_features_list, folds = 1000)

########################################################## Gridsearch

from sklearn.model_selection import StratifiedShuffleSplit

def get_param(key):

    param_grid_dict = { 'SVC':         {'clf__C':               [1, 50, 100, 200, 300, 500, 1e3, 5e3, 1e4, 5e4, 1e5],
                                        'clf__gamma':           ['scale','auto', 0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1],
                                        'clf__class_weight':    ['balanced', None]},
                        'DecisionTree': {'clf__min_samples_split':  [2,3,4,5,6,7],
                                        'clf__criterion':           ['gini', 'entropy'],
                                        'clf__max_features':        [None, 1,2,3,4],
                                        'clf__class_weight':        ['balanced', None]},
                        'GaussianNB':   {},
                                        }
 
    return param_grid_dict[key]

######

def clf_opt(key,clf,features,labels):

    param_grid = get_param(key)
    
    scaler = MinMaxScaler()

    pipe = Pipeline([('scaler', scaler), ('clf', clf)])

    cv = StratifiedShuffleSplit(n_splits=100,random_state=42)

    search = GridSearchCV(pipe, param_grid,  cv = cv, scoring = 'f1')

    features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.10, random_state=42, stratify=labels)
    search.fit(features_train, labels_train)

    print ('Proportion of poi in label_test : ',round(sum(labels_test)/len(labels_test),3))
    print ('Proportion of poi in label_train : ',round(sum(labels_train)/len(labels_train),3))
    print ('')
    print ('Grid best parameter: ', search.best_params_)
    print ('Grid best score: ', search.best_score_)
    print ('Grid best estimator: ',search.best_estimator_.named_steps['clf'])
    print ('')
    clf = search.best_estimator_.named_steps['clf']
    scaler = search.best_estimator_.named_steps['scaler']

    features_test = scaler.transform(features_test)
    labels_pred = clf.predict(features_test)
    print ('Validate with test set.')
    print ('')
    print ('Validate accuracy: ',round(accuracy_score(labels_test, labels_pred),3))
    print ('Validate precision: ',round(precision_score(labels_test, labels_pred),3))
    print ('Validate recall: ',round(recall_score(labels_test, labels_pred),3))
    print ('Validate f1: ',round(f1_score(labels_test, labels_pred),3))

data = featureFormat(my_dataset, selected_features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)

clf = SVC(C=1000, gamma='auto')
key = 'SVC'

#clf_opt(key,clf,features,labels)