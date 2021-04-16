#!/usr/bin/python
from __future__ import division, print_function
import sys
import pickle
import pprint
import pandas as pd

import matplotlib
matplotlib.use('TKAgg')
import matplotlib.pyplot as plt

import seaborn as sns
sns.set()

sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
import tester as tester

from sklearn.feature_selection import SelectKBest, chi2, f_classif
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedShuffleSplit
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix, f1_score

from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import linear_model
from sklearn.cluster import KMeans

from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline

### Load the dictionary containing the dataset
def load_dataset():
    with open("final_project_dataset.pkl", "r") as data_file:
        data_dict = pickle.load(data_file)
    return data_dict

### Check dataset
def check_dataset(data_dict):
    
    nan_all_list =[]
    nan_email_list = []
    poi_data_dict ={}

    for data_name in data_dict.keys():
        
        if data_dict[data_name]['email_address']=="NaN":
            nan_email_list.append(data_name)
        if data_dict[data_name]['poi']==True:
            poi_data_dict[data_name]=data_dict[data_name]

        count = 0
        for feature in data_dict[data_name]:
            check_list = ["NaN", True, False]
            if data_dict[data_name][feature] in check_list:
                count+= 1
        if count == len(data_dict[data_name].keys()):
            nan_all_list.append(data_name)

    return (nan_email_list, nan_all_list, poi_data_dict)

### Check for NaNs
def nan_check(data_dict):
    nan_dict = {}

    #print len(data_dict.keys())

    for data_name in data_dict.keys():
        for data_point in data_dict[data_name]:
            nan_dict.setdefault(data_point,0)
            if data_dict[data_name][data_point]=="NaN":
                nan_dict[data_point]+=1

    df = pd.DataFrame.from_dict(nan_dict, orient='index', columns =["NaN Count"])

    df["NaN Proportion"]=df["NaN Count"]/len(data_dict)
    
    df = df.sort_values("NaN Proportion", ascending=True)
    
    return df

### Prepare Dataframe for Plots v1
def df_features(data_dict,feature_list, remove_all_zeroes=False, remove_any_zeroes=False,):
    
    temp_dict = {}
    
    for item in data_dict.keys():
        
        temp_list = []
        
        for feature in feature_list:
            
            value = data_dict[item][feature]

            if value == "NaN":
                value = 0
            
            temp_list.append(value)

        if remove_all_zeroes:

            if 0 in temp_list:
                
                continue
            else:
                
                temp_dict.setdefault(item,temp_list)
        else:
            
            temp_dict.setdefault(item,temp_list)

    df = pd.DataFrame.from_dict(temp_dict, orient='index', columns =feature_list)
    
    return df

### Extract Featurelist
def get_feature(data_dict, excluded_features):
    
    feature_dict = {}

    for data_point in data_dict.keys():
        for feature in data_dict[data_point]:
            if feature not in excluded_features and feature != 'poi':
                feature_dict.setdefault(feature,0)

    feature_list = feature_dict.keys()
    feature_list.sort()
    feature_list = ['poi']+feature_list

    return feature_list

### Check Outlier
def check_outlier(data_dict, feature):
    result_list=[]
    for data_point in data_dict.keys():
        value = data_dict[data_point][feature]
        poi_status = data_dict[data_point]['poi']
        if value !="NaN":
            result_list.append([value, data_point, poi_status])

    result_list.sort(reverse = True)

    print (pd.DataFrame(result_list[0:5],columns=[feature,'Name','Poi']))

### Remove Outlier
def remove_outlier(data_dict, items):
    
    new_dict = {}
    
    for data_point in data_dict.keys():
        if data_point in items:
            continue
        else:
            new_dict[data_point]=data_dict[data_point]

    return new_dict
    
def computeFraction( poi_messages, all_messages ):

    if poi_messages == "NaN" or all_messages == "NaN":
        fraction = 0.
    else:
        fraction = poi_messages*1.0/all_messages

    return fraction

def create_new_message_feature(data_dict):

    new_dict = {}
    
    for data_point in data_dict:

        new_dict[data_point]=data_dict[data_point]

        feature_dict = new_dict[data_point]

        from_poi_to_this_person = feature_dict["from_poi_to_this_person"]
        to_messages = feature_dict["to_messages"]
        feature_dict["fraction_from_poi"] = computeFraction( from_poi_to_this_person, to_messages )

        from_this_person_to_poi = feature_dict["from_this_person_to_poi"]
        from_messages = feature_dict["from_messages"]
        feature_dict["fraction_to_poi"] = computeFraction( from_this_person_to_poi, from_messages )

    return new_dict
    
def get_KBest(features, labels, k, feature_list):
    selector = SelectKBest(f_classif, k = k)
    kBest_features = selector.fit_transform(features, labels)

    mask = selector.get_support() #list of booleans
    new_features = [] # The list of your K best features
    new_features_scores = [] # The list of your K best scores

    for bool, feature, score in zip(mask, feature_list[1:], selector.scores_):
        if bool:
            new_features.append(feature)
            new_features_scores.append(score)

    scores = zip(new_features,new_features_scores)

    sorted_scores = sorted(scores, key = lambda x: x[1], reverse=True)

    kBest_feature_list = [i[0] for i in sorted_scores]

    return (kBest_features, sorted_scores, kBest_feature_list)

def clf_func(k, clf_dict, sorted_feature_list, my_dataset, clf_key):

    result_dict= {}
    k_result_dict={}
    result_list=[]

    for ii in k:

        selected_features_list = sorted_feature_list[0:ii+1]

        data = featureFormat(my_dataset, selected_features_list, sort_keys = True)
        labels, features = targetFeatureSplit(data)

        for key in sorted(clf_dict):
        
            clf = clf_dict[key]

            true_negatives = 0
            false_negatives = 0
            true_positives = 0
            false_positives = 0

            for i in range(0,1000):
        
                features_train, features_test, labels_train, labels_test = train_test_split(features, labels, test_size=0.1, random_state=i)

                pipe = Pipeline([('scaler', MinMaxScaler()), ('clf', clf)])
                
                pipe.fit(features_train, labels_train)

                pred = pipe.predict(features_test)

                for j, prediction in enumerate(pred):
                
                    test_label = labels_test[j]
                
                    if prediction == 0 and test_label == 0:
                        true_negatives += 1
                    elif prediction == 0 and test_label == 1:
                        false_negatives += 1
                    elif prediction == 1 and test_label == 0:
                        false_positives += 1
                    elif prediction == 1 and test_label == 1:
                        true_positives += 1
                    else:
                        print ("Warning: Found a predicted label not == 0 or 1.")
                        print ("All predictions should take value 0 or 1.")
                        print ("Evaluating performance for processed predictions:")
                        break
        
            try:
                total_predictions = true_negatives + false_negatives + false_positives + true_positives
                accuracy = 1.0*(true_positives + true_negatives)/total_predictions
                precision = 1.0*true_positives/(true_positives+false_positives)
                recall = 1.0*true_positives/(true_positives+false_negatives)
                f1 = 2.0 * true_positives/(2*true_positives + false_positives+false_negatives)
                f2 = (1+2.0*2.0) * precision*recall/(4*precision + recall)
                
                accuracy = round(accuracy,4)
                precision = round(precision,4)
                recall = round(recall,4)
                f1 = round(f1,4)
                f2 = round(f2,4)

                result_dict.setdefault(ii,{})
                result_dict[ii].setdefault(key,[accuracy, precision, recall, f1, total_predictions, true_positives, false_positives, false_negatives , true_negatives])
            
            except:
                print ("Got a divide by zero when trying out:", clf)
                print ("Precision or recall may be undefined due to a lack of true positive predicitons.")

            if key == clf_key:
                
                result_list.append([f1,ii,selected_features_list, pipe])

                result_list.sort(reverse=True)

                output = result_list[0]

    result_metrics = ['accuracy','precision',' recall','f1','total_predictions',' tp','fp','fn','tn']

    
    for ii in k:
        k_result_dict[ii] = result_dict[ii][clf_key]
    
    df_k_result = pd.DataFrame.from_dict(k_result_dict, orient='index', columns =result_metrics)
    df_k_result = df_k_result.sort_index(axis = 0)
    
    output.append(df_k_result)

    df_k_best_result = pd.DataFrame.from_dict(result_dict[output[1]], orient='index', columns =result_metrics)
    df_k_best_result = df_k_best_result.sort_index(axis = 0)

    output.append(df_k_best_result)

    return output

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

def main():
    ### load dataset
    data_dict = load_dataset()

    print ("Datapoints:",len(data_dict))
    print ("Features:",len(data_dict['BELDEN TIMOTHY N']))

    # Check features
    feature_list = get_feature(data_dict, [])
    df = pd.DataFrame.from_dict(data_dict, orient='index', columns =feature_list )
    mask = df['poi']==True

    print ("Person of Interest (POI):",len(df[mask]['poi']))
    print ('')
    print ('List of POIs:')
    print ('')
    print (df[mask]['poi'])

    print (nan_check(data_dict))

    feature_list = ['poi', 'total_stock_value', 'total_payments']
    df=df_features(data_dict,feature_list)
    sns.pairplot(df, hue='poi',height=4)
    plt.show()

    # Identify outlier
    check_outlier(data_dict, 'total_payments')
    print ('')

    # Check data for additional problematic datapoints
    nan_email_list, nan_all_list, poi_data_dict = check_dataset(data_dict)
    print (nan_all_list)

    # Remove identified problematic datapoints
    outlier = ['TOTAL', 'THE TRAVEL AGENCY IN THE PARK', 'LOCKHART EUGENE E']
    my_dataset = remove_outlier(data_dict, outlier)

    # Create new features
    my_dataset = create_new_message_feature(my_dataset)

    # Exclude non-relevant features and get final feature List
    excluded_features   = ['email_address']
    feature_list        = get_feature(my_dataset, excluded_features)

    # Split data in labels and features
    data = featureFormat(my_dataset, feature_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    k=len(features[0])

    # Get sorted features according the KBest scores
    kBest_features, sorted_scores, kBest_features_list = get_KBest(features, labels, k, feature_list)

    sorted_features_list = ['poi'] + kBest_features_list

    print (pd.DataFrame(sorted_scores,columns=['Feature','KBest Score']))


    feature_list = ['poi', 'exercised_stock_options', 'total_stock_value']
    df=df_features(my_dataset,feature_list)
    sns.pairplot(df, hue='poi',height=4)
    plt.show()

    # Dictionary with different classifiers
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

    #k =[6]

    clf_key = 'SVC'

    # Main function
    ## Input:
    ## [k] = list of different features quantity
    ## clf_dict = dictionary with different classifiers
    ## sorted_features_list = ranked features according KBest scores (complete)
    ## my_dataset = modified data_dict
    f1, k_best_features_number, selected_features_list, pipe, df_k_result, df_k_best_result = clf_func(k, clf_dict, sorted_features_list, my_dataset, clf_key)

    print ('')
    print ('Result Summary for K-Features')
    print ('')
    print (df_k_result)
    print ('')
    print ('Best Results for ', k_best_features_number, ' features')
    print ('')
    print ('Selected Fetaures: ',len(selected_features_list[1:]), selected_features_list[1:])
    print ('')
    print (df_k_best_result)
    print ('')

    df_k_result.plot(kind='bar', y='f1', figsize=(9,5))
    plt.title('SVC F1 vs. K-Features')
    plt.xlabel('K-Feature')
    plt.ylabel('F1-Score')
    plt.show()

    data = featureFormat(my_dataset, selected_features_list, sort_keys = True)
    labels, features = targetFeatureSplit(data)

    clf = SVC(C=1000, gamma='auto')
    key = 'SVC'

    clf_opt(key,clf,features,labels)


    acc, pre, rec, f1, total, tp, fp, fn, tn = df_k_result.loc[k_best_features_number].to_numpy()

    print ('')
    print ('True Positives (tp) :', tp)
    print ('False Positives (fp) :', fp)
    print ('False Negatives (fn) :', fn)
    print ('True Negatives (tn) :', tn)
    print ('')
    print ('Accuracy: ', (tp+tn)/(tp+fp+tn+fn))
    print ('Prescision: ', tp/(tp+fp))
    print ('Recall: ', tp/(tp+fn))

    tester.dump_classifier_and_data(pipe, my_dataset, selected_features_list)

    # Load pkl files for evaluation & run testing script
    tester.test_classifier(pipe, my_dataset, selected_features_list, folds = 1000)

    ### Run testing script
    #test_classifier(clf, dataset, feature_list)

if __name__ == '__main__':
    main()