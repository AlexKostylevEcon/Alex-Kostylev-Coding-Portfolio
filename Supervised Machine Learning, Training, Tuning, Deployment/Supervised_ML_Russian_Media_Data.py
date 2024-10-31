# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 20:39:20 2024

@author: Alex

This code trains and deploys the classification model whether the media article mentions governors engaging with the federal center
"""

# Importing relevant packages

import os
import re
import pandas as pd
import numpy as np
from numpy import mean
from numpy import std
import time
from pprint import pprint
from datetime import datetime

from sklearn.ensemble import RandomForestRegressor, LogisticRegression, RandomForestClassifier, XGBClassifier, StackingClassifier, LinearSVC
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.model_selection import KFold, cross_val_score
from sklearn.feature_selection import SelectFpr, chi2, SelectKBest, SelectFwe, f_classif, SelectFdr
from sklearn.pipeline import make_pipeline
from sklearn.metrics import classification_report, roc_curve, precision_recall_curve, auc, make_scorer, recall_score, accuracy_score, precision_score, confusion_matrix, f1_score
from sklearn.model_selection import GridSearchCV, StratifiedKFold, RandomizedSearchCV

from multiprocessing.pool import ThreadPool

from NLP_preprocessing import preprocessing

def grid_search(model, refit_score):
    
    skf = StratifiedKFold(n_splits=5)
    grid_search = GridSearchCV(clf, param_grid, scoring=scorers, refit=refit_score, cv=skf, return_train_score=True, n_jobs=12, verbose=1)
    
    grid_search.fit(X_train, y_train)

    # make the predictions
    y_pred = grid_search.predict(X_test)

    print('Best params for {}'.format(refit_score))
    print(grid_search.best_params_)

    # confusion matrix on the test data.
    print('\nConfusion matrix of Random Forest optimized for {} on the test data:'.format(refit_score))
    print(pd.DataFrame(confusion_matrix(y_test, y_pred),
                 columns=['pred_neg', 'pred_pos'], index=['neg', 'pos']))
    return grid_search


# Reading data (the data cannot be shared due to the licensing agreement)
# Reading Governors Meta data

Governors_table = pd.read_csv("C:/Users/Alex/Desktop/Russian data. Increasing accuracy/All governors list.csv")

Governornames=Governors_table["Search Inquiry"].tolist()
entrydates=Governors_table["Entry"].tolist()
enddates=Governors_table["End"].tolist()
Region=Governors_table["Region"].tolist()

# Reading dictionaries

Representative_dictionary_table = pd.read_csv("C:/Users/Alex/Desktop/Russia. Increasing accuracy/RepresentativeDictionary.csv")
Representative_dict=Representative_dictionary_table["Word"]

Federal_dictionary_table = pd.read_csv("C:/Users/Alex/Desktop/Russia. Increasing accuracy/Federaldictionary.csv")
Federal_dict=Federal_dictionary_table["Word"]

ru_cities=pd.read_excel("C:/Users/Alex/Desktop/Russia. Increasing accuracy/Russian Cities.xlsx")
foreign_cities=pd.read_excel("C:/Users/Alex/Desktop/Russia. Increasing accuracy/Foreign Cities.xlsx")
foreign_countries=pd.read_excel("C:/Users/Alex/Desktop/Russia. Increasing accuracy/Foreign Countries.xlsx")

foreign_cities_dict=foreign_cities["Города"]
foreign_countries_dict=foreign_countries["Country"]
ru_capitals=ru_cities["Capital"]

midfeddict=pd.read_excel("C:/Users/Alex/Desktop/Russia. Increasing accuracy/Mid federal officials names.xlsx")
midfeddict=midfeddict["Mid Federal Officials"].tolist()
for x in range(len(Governornames)):
    Governornames[x]=re.sub(r'\(', '', Governornames[x])
    Governornames[x]=Governornames[x].split(" ",1)[0]
    
dictionaries=pd.read_excel("C:/Users/Alex/Desktop/Russia. Increasing accuracy/Policy Direction Lists.xlsx")

econdict=dictionaries["Economy"]
socialdict=dictionaries["Socio-cultural"]    
    

# Reading texts 

Texts_table_1 = pd.read_excel("C:/Users/Alex/Desktop/Russia. Increasing accuracy/fedeal problems combined.xlsx")

datelist=[]
indexlist=[]
texts=[]
substring="Дата выпуска:"

text=Texts_table_1["text"]

# Extracting date of publication from texts

for i in range(len(text)):
    if substring in text[i]:
      date1 = re.search(r'\d{2}.\d{2}.\d{4}', text[i])
      try:
         dateform=datetime.strptime(date1.group(), '%d.%m.%Y').date()
      except:
         if date1 is None:
           print("date None")
    if date1 is not None:
          texts.append(text[i])
          indexlist.append(i)
          datelist.append(dateform) 

Texts_table_1["date"]=datelist

# Changing labels for articles that were relabeled after 2nd annotation

Texts_table_1["Federal Officials"]=Texts_table_1["label"]
Texts_table_1.loc[Texts_table_1['Revisited'] ==1, 'Federal Officials'] = 1
Texts_table_1.loc[Texts_table_1['Revisited'] ==0, 'Federal Officials'] = 0

# Dropping rows with no text 

Texts_table_1 = Texts_table_1[Texts_table_1.text == Texts_table_1.text]

# Merging with the second training data file

Texts_table_2=pd.read_excel("C:/Users/Alex/Desktop/Russia. Increasing accuracy/human in loop 2008 2.xlsx")
Texts_table_2 = Texts_table_2[Texts_table_2.text == Texts_table_2.text]
Texts_table_2=Texts_table_2.reset_index(drop=True)
Texts_table_2["Federal Officials"]=Texts_table_2["Federal Officials"].fillna(0)
Texts_table_2 = Texts_table_2[Texts_table_2["Federal Officials"]==Texts_table_2["Federal Officials"]]

Merged_text_table=pd.concat([Texts_table_1, Texts_table_2], ignore_index=True)

# Keeping only relevant columns

collist=["governor", "Federal Officials", "date", "About Appointment", "text", "2002"]
Merged_text_table=Merged_text_table[collist]

Merged_text_table["About Appointment"]=Merged_text_table["About Appointment"].fillna(0)

# Dropping texts that are not annotated
Merged_text_table = Merged_text_table[Merged_text_table["Federal Officials"]==Merged_text_table["Federal Officials"]]

# Dropping duplicates
Merged_text_table=Merged_text_table.drop_duplicates(subset=['text', 'governor'])

Merged_text_table.loc[Merged_text_table['About Appointment'] ==1, 'Federal Officials'] = 1

print("Articles directly mentioning appointment:", sum(Merged_text_table["About Appointment"]))
print("Articles mentioning engagement with federal officials:", sum(Merged_text_table["Federal Officials"]))


Merged_text_table=Merged_text_table.reset_index(drop=True)

governors=Merged_text_table["governor"].values.tolist()
datelist=Merged_text_table["date"].tolist()
federallist=Merged_text_table["Federal Officials"].values.tolist()
texts=Merged_text_table["text"].values.tolist()
datelist=Merged_text_table["date"].values.tolist()

# Generating governor's region variable for each article

region=[]
for x in range(len(governors)):
  for l in range(len(Governornames)): 
    if governors[x]==Governornames[l]:
       region.append(Region[l])
       if l==len(Governornames)-1:
          print(governors[x])
       break

# Generating governor's regional capital variable for each article

reg_capital=[]
for x in range(len(region)):
    for index in ru_cities.index:
        if region[x]==ru_cities["Region"][index]:
            reg_capital.append(ru_cities["Capital"][index])
          
# Running the preprocessing and feature engineering function, multithreading

tic = time.perf_counter()

inputs=[(texts, governors, reg_capital, ru_capitals, foreign_cities_dict, midfeddict, foreign_countries_dict, Representative_dict, Federal_dict, datelist)]
pool = ThreadPool(8)
b = pool.starmap(preprocessing, inputs)        

toc = time.perf_counter()
print(toc - tic)
        
frames =b[0][1]
featureframe = frames

governorall=b[0][2]
datelist1= b[0][3]
rawtexts1=b[0][4]
cleansentences=b[0][0]
federal_labels=b[0][5]

labels_frame = pd.Dataframe(np.asarray(federal_labels).reshape(-1, 1))

oldcleansentences=cleansentences

d1=pd.DataFrame(cleansentences)
d2=featureframe

## Training the classification model
# Separate Train and Test parts of the old dataset. 

Vectorizer = CountVectorizer(min_df=0.005, max_df=0.7, ngram_range=(1,2))
Bag = Vectorizer.fit_transform(cleansentences).toarray()

Tfidfconverter = TfidfTransformer()
Bag = Tfidfconverter.fit_transform(Bag).toarray()

BagFeatures=np.concatenate((Bag, featureframe), axis=1)
BagFeaturesframe=pd.DataFrame(BagFeatures)

X_train, X_test, y_train, y_test = train_test_split(BagFeaturesframe, labels_frame, test_size=0.1, random_state=0, stratify = labels_frame)

# Baseline random forest performance

base_model = RandomForestClassifier(n_estimators=1000, class_weight='balanced', random_state=0, oob_score = True, bootstrap = True, max_depth=4)
base_model.fit(X_train, y_train) 
y_pred_prob = base_model.predict_proba(X_test)[:, 1]
y_pred=np.where(y_pred_prob>0.5, 1, 0)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))

# Determining feature importance and keeping only the 1000 most important features

bag_names = Vectorizer.get_feature_names_out()
feature_names=featureframe.columns.to_numpy()
BagFeatures_names=(np.concatenate((bag_names, feature_names),axis=0)).tolist()
BagFeaturesframe = pd.DataFrame(BagFeatures, columns=[BagFeatures_names])

k = 1000
selector=SelectKBest(f_classif, k=k)

X_new = SelectKBest(selector).fit_transform(BagFeaturesframe, labels_frame)
selected_features = BagFeaturesframe.columns[SelectKBest(f_classif, k=k).fit(BagFeaturesframe, labels_frame).get_support()]


# Tuning the random forest model

random_state=[0]
class_weight=["balanced"]
# Number of trees in random forest
n_estimators = [int(x) for x in np.linspace(start = 600, stop = 2000, num = 10)]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [int(x) for x in np.linspace(20, 80, num = 4)]
max_depth.append(None)
# Minimum number of samples required to split a node
min_samples_split = [int(x) for x in np.linspace(start = 5, stop = 50, num = 10)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [2, 4, 8]
# Method of selecting samples for training each tree
bootstrap = [True, False]
# Create the random grid
param_grid = {"random_state":random_state,
              'class_weight': class_weight,
              'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
pprint(param_grid)

#'max_features': max_features,

scorers = {
    'precision_score': make_scorer(precision_score),
    'recall_score': make_scorer(recall_score),
    'accuracy_score': make_scorer(accuracy_score),
    'f1':  make_scorer(f1_score)
}

start = time.time()

rf_base_model = RandomForestClassifier(n_jobs=12, random_state=0)

grid_search_clf = grid_search(rf_base_model, refit_score='f1')

results = pd.DataFrame(grid_search_clf.cv_results_)
results = results.sort_values(by='mean_test_precision_score', ascending=False)
results1=results[['mean_test_precision_score', 'mean_test_recall_score', 'mean_test_accuracy_score', 'mean_test_f1', 'param_n_estimators', 'param_max_features', 'param_max_depth', 'param_min_samples_split','param_min_samples_leaf','param_bootstrap']].round(3)

end = time.time()
print(end - start)

# Running hypertuned random forest, best f1

X_train, X_test, y_train, y_test = train_test_split(X_new, labels_frame, test_size=0.2, random_state=0, stratify = labels_frame)

best_random_f = RandomForestClassifier(n_estimators=2000, class_weight='balanced', random_state=0, oob_score = True, bootstrap = True, max_depth=None, \
                                    min_samples_split=45, min_samples_leaf=2, max_features="sqrt")
best_random_f.fit(X_train, y_train) 
y_pred_prob = best_random_f.predict_proba(X_test)[:, 1]
y_pred=np.where(best_random_f>0.5, 1, 0)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))

f1=[]
f1.append(f1_score(y_test,y_pred,average="binary"))

cv = KFold(n_splits=10, random_state=1, shuffle=True)

# evaluate model with cross valuation based on f1, precision and recall
scores = cross_val_score(best_random_f, X_new, labels_frame, scoring='f1', cv=cv, n_jobs=-1)
scores2 = cross_val_score(best_random_f, X_new, labels_frame, scoring='precision', cv=cv, n_jobs=-1)
scores3 = cross_val_score(best_random_f, X_new, labels_frame, scoring='recall', cv=cv, n_jobs=-1)

print('f1: %.3f (%.3f)' % (mean(scores), std(scores)))
print('precision: %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('recall: %.3f (%.3f)' % (mean(scores3), std(scores3)))

## Preparing models for stacking: ensamble of 3 random forest models (highest f1, precision and recall respectively), XGboost and logit
# Stack 3 random forests: highest f1, highest precision, highest recall. Then add xgb, logit and svr    

#2. Random forest with the highest precision

precision_random_f = RandomForestClassifier(n_estimators=2000, class_weight='balanced', random_state=0, oob_score = True, bootstrap = True, max_depth=None, \
                                    min_samples_split=2, min_samples_leaf=1, max_features="sqrt")

precision_random_f.fit(X_train, y_train) 
y_pred_prob = precision_random_f.predict_proba(X_test)[:, 1]
y_pred=np.where(y_pred_prob>0.5, 1, 0)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))
    
#3. Random forest with the highest recall

recal_random_f = RandomForestClassifier(n_estimators=400, class_weight='balanced', random_state=0, oob_score = True, bootstrap = True, max_depth=10, \
                                    min_samples_split=10, min_samples_leaf=4, max_features=None)

recal_random_f.fit(X_train, y_train) 
y_pred_prob = recal_random_f.predict_proba(X_test)[:, 1]
y_pred=np.where(y_pred_prob>0.5, 1, 0)
print(confusion_matrix(y_test,y_pred))
print(classification_report(y_test, y_pred))

# 4. XGboost classifier

xgb_clf = XGBClassifier(
                        scale_pos_weight=10, 
                        objective='binary:logistic', 
                        nthread=-1, 
                        max_depth=7, 
                        )
xgb_clf.fit(X_train, y_train)

# evaluate model

cv = KFold(n_splits=10, random_state=1, shuffle=True)

scores = cross_val_score(xgb_clf, X_new, labels_frame, scoring='f1', cv=cv, n_jobs=-1)
scores2 = cross_val_score(xgb_clf, X_new, labels_frame, scoring='precision', cv=cv, n_jobs=-1)
scores3 = cross_val_score(xgb_clf, X_new, labels_frame, scoring='recall', cv=cv, n_jobs=-1)

print('f1: %.3f (%.3f)' % (mean(scores), std(scores)))
print('precision: %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('recall: %.3f (%.3f)' % (mean(scores3), std(scores3)))

# Stacking classifier

estimators = [
     ('rf_best', best_random_f),
     ("rf_precision", precision_random_f),
     ("rf_recall", recal_random_f),
  ('xgb', xgb_clf),
  ('svr', make_pipeline(StandardScaler(), LinearSVC())),
  ('lr',LogisticRegression())]

clf = StackingClassifier(estimators=estimators, final_estimator=LogisticRegression())

print(clf.fit(X_train, y_train).score(X_test, y_test))

# evaluate model

cv = KFold(n_splits=10, random_state=1, shuffle=True)

scores = cross_val_score(clf, X_new, labels_frame, scoring='f1', cv=cv, n_jobs=-1)
scores2 = cross_val_score(clf, X_new, labels_frame, scoring='precision', cv=cv, n_jobs=-1)
scores3 = cross_val_score(clf, X_new, labels_frame, scoring='recall', cv=cv, n_jobs=-1)

print('f1: %.3f (%.3f)' % (mean(scores), std(scores)))
print('precision: %.3f (%.3f)' % (mean(scores2), std(scores2)))
print('recall: %.3f (%.3f)' % (mean(scores3), std(scores3)))


#### Applying the best classification model to the full text dataset

featurestablefull=pd.DataFrame()
texttablefull=pd.DataFrame()

for i in range(1,14):
    
    featurestable1 = pd.read_excel("C:/Users/Alex/Desktop/Russia. Increasing accuracy/Full preprocess/full_federal_features"+str(i)+".xlsx")
    texttable1= pd.read_excel("C:/Users/Alex/Desktop/Russia. Increasing accuracy/Full preprocess/full_federal_pre-text"+str(i)+".xlsx")

    featurestablefull=pd.concat((featurestablefull,featurestable1), axis=0)
    texttablefull=pd.concat((texttablefull, texttable1), axis=0)

frames=[]
    
rawtextfull = pd.read_excel("C:/Users/Alex/Desktop/Russia. Increasing accuracy/Full preprocess raw/full_federal_pre-rawtext1.xlsx")

for p in range(1,14):
     rawtexttable1 = pd.read_excel("C:/Users/Alex/Desktop/Russia. Increasing accuracy/Full preprocess raw/full_federal_pre-rawtext"+str(p)+".xlsx")
     frames.append(rawtexttable1)

rawtextfull = pd.concat(frames)

datefull=[]
y_pred_fullmajor=[]
y_pred_prob_fullmajor=[]
governorfull=[]
textfull=[]

featurestablefull=featurestablefull.drop("Unnamed: 0", axis=1)

featurestablefull=featurestablefull.reset_index(drop=True)
featurestablefull['index'] = featurestablefull.index

texttablefull=texttablefull.reset_index(drop=True)
texttablefull['index'] = texttablefull.index

rawtextfull=rawtextfull.reset_index(drop=True)
rawtextfull['index'] = rawtextfull.index


#2. Delete duplicates
### Create a list for each governor
### Add to list of lists and loop through each list to keep only unique
### Merge into one list again.

governorset = set(rawtextfull["governor"])

frames=[]

for governor in governorset:
    thisgovernortexts=[]
    thisgovernortexts=rawtextfull.loc[rawtextfull['governor'] == str(governor)]
    thisgovtextsset=set(thisgovernortexts['clean text'])
    thisgovernortexts=thisgovernortexts.drop_duplicates(subset=['clean text'])
    
    frames.append(thisgovernortexts)

textsnoduplicates = pd.concat(frames)
    
indexset=set(textsnoduplicates['index'])

featurestablefull=featurestablefull.loc[featurestablefull.index.isin(indexset)]
rawtextfull=rawtextfull.loc[rawtextfull.index.isin(indexset)]
texttablefull=texttablefull.loc[texttablefull.index.isin(indexset)]


rawtextfull=rawtextfull.drop("index", axis=1)
featurestablefull=featurestablefull.drop("index", axis=1)
textsnoduplicates=textsnoduplicates.drop("index", axis=1)

cleansentences=texttablefull["clean text"].tolist()
governorlist1=texttablefull["governor"]
datelist1=texttablefull["date"]
rawtexts=rawtextfull["clean text"]


####################
## Isolating interviews

interview=[]
for i in range(len(cleansentences)):
    interview.append(0)
    s=cleansentences[i]
    
    if ((re.search(r'\b'+"интервью", s.lower())) or \
        ((re.search(r'\b'+"задать", s.lower())) or (re.search(r'\b'+"ответить", s.lower())) or (re.search(r'\b'+"задавать", s.lower())) or \
         (re.search(r'\b'+"отвечать", s.lower())) and (re.search(r'\b'+"вопрос", s.lower()))) or \
         ((re.search(r'\b'+"беседа", s.lower())) or (re.search(r'\b'+"собеседник", s.lower())) or (re.search(r'\b'+"беседовать", s.lower())) or \
          (re.search(r'\b'+"побеседовать", s.lower()))) and \
           (re.search(r'\b'+"searchword", s.lower()))):
                interview[i]=1  


####################
# Classifying the whole dataset in batches

n=len(cleansentences)
m=100

f=round(n/m)
print(f)
i=0

for i in range(0,m):
 
    cleaninterval=cleansentences[(f*(i)):(f*(i+1))]
    featureinterval=featurestablefull[(f*(i)):(f*(i+1))]
 
    FullBag=Vectorizer.transform(cleaninterval)
    FullBag = Tfidfconverter.transform(FullBag).toarray()
    
    BagFeaturesFull=np.concatenate((FullBag, featureinterval), axis=1)
    
    BagFeaturesFull=selector.transform(BagFeaturesFull)
    
    y_pred_prob = base_model.predict_proba(BagFeaturesFull)[:,1]
    y_pred=[1 if y_prob>=0.5 else 0 for y_prob in y_pred_prob]
    
    y_pred_fullmajor.extend(y_pred)
    y_pred_prob_fullmajor.extend(y_pred_prob)
 
y_class_prob=[]    
 
for i in range(0, len(y_pred_fullmajor)):
    if y_pred_fullmajor[i]==1:
        y_class_prob.append(y_pred_prob_fullmajor[i])
    else:
        y_class_prob.append(0)

dnew = {'Interview':interview, 'Other Governors': featurestablefull["governoryeslist"], 'With President': featurestablefull["fedearlinq1"], 'positives': featurestablefull["positive"], 'negatives': featurestablefull["negative"],'governor':governorlist1, 'date':datelist1, 'rawtext': rawtexts, 'clean text': cleansentences, 'federal predicted': y_pred_fullmajor, 'federal predicted probability': y_class_prob}
    
dfnew = pd.DataFrame(data=dnew)

dfnew["no other governors"]=np.where((dfnew["Other Governors"]==1), 0,1)
dfnew["federal with president"]=np.where(((dfnew['federal predicted']==1) & (dfnew["With President"]==1)), 1,0)
dfnew["federal without others"]=np.where(((dfnew['federal predicted']==1) & (dfnew["Other Governors"]==0)), 1,0)

dfnew["federal postive"]=np.where(((dfnew['federal predicted']==1)),dfnew["positives"],0)
dfnew["federal negative"]=np.where(((dfnew['federal predicted']==1)),dfnew["negatives"], 0)

dfnew["President postive"]=np.where((dfnew['federal with president']==1),dfnew["positives"],0)
dfnew["President negative"]=np.where((dfnew['federal with president']==1),dfnew["negatives"], 0)

dfnew["interview federal"]=np.where(((dfnew['federal predicted']==1) & (dfnew["Interview"]==1)),1, 0)
dfnew["no other governors interview"]=np.where(((dfnew['Other Governors']==0) & (dfnew["Interview"]==1)),1, 0)
dfnew["no other governors federal interview"]=np.where(((dfnew['Other Governors']==0) & (dfnew["Interview"]==1) & (dfnew['federal predicted']==1)),1, 0)

dfnew["Month"]=pd.DatetimeIndex(dfnew['date']).month
dfnew["Year"]=pd.DatetimeIndex(dfnew['date']).year

#4. Generating the lists of publishers

mainsources2012=pd.read_excel("C:/Users/Alex/Desktop/Russian Paper/Publisher Lists/2012 list.xlsx")
mainpublishers2012=mainsources2012["sources"].tolist()

mainsources2010=pd.read_excel("C:/Users/Alex/Desktop/Russian Paper/Publisher Lists/2010 list.xlsx")
mainpublishers2010=mainsources2010["sources"].tolist()

governors=governorlist1

sources=[]

for t in rawtexts:
    sources.append(t.partition('\n')[0])

dictionary={"source": sources , "governor": governors}
sourcetable=pd.DataFrame.from_dict(dictionary)

# Loop through texts, if publisher in list - add 1 to the relevant list; otherwise - 0
sourcetable['2012 list'] = 0
sourcetable.loc[sourcetable['source'].isin(mainpublishers2012), '2012 list'] = 1

print(sum(sourcetable['2012 list']))

sourcetable['2010 list'] = 0
sourcetable.loc[sourcetable['source'].isin(mainpublishers2010), '2010 list'] = 1

print(sum(sourcetable['2010 list']))

dfnew['2010 list']=sourcetable['2010 list']
dfnew['2012 list']=sourcetable['2012 list']

list1=np.array_split(dfnew, 2)

os.makedirs("C:/Users/Alex/Desktop/federaldatasetOctober2023", exist_ok=True)

writer = pd.ExcelWriter("C:/Users/Alex/Desktop/federaldatasetOctober2023/federal all texts all sources clean dup 1.xlsx", engine='xlsxwriter')
writer.book.use_zip64()
list1[0].to_excel(writer, merge_cells=False)
writer.close()

writer = pd.ExcelWriter("C:/Users/Alex/Desktop/federaldatasetOctober2023/federal all texts all sources clean dup 2.xlsx", engine='xlsxwriter')
writer.book.use_zip64()
list1[1].to_excel(writer, merge_cells=False)
writer.close()

os.makedirs("C:/Users/Alex/Desktop/federaldatasetOctober2023", exist_ok=True)

writer = pd.ExcelWriter("C:/Users/Alex/Desktop/federaldataset/federaltest all sources text.xlsx", engine='xlsxwriter')
writer.book.use_zip64()
dfnew[0:10000].to_excel(writer, merge_cells=False)
writer.close()

#5. Aggregate by governor, months and sources

dfnew["Quarter"]=pd.DatetimeIndex(dfnew['date']).quarter

obsnum=dfnew.groupby(["Year","Quarter", "governor"]).size()

meeting=[]
i=0

for text in cleansentences:
    meeting.append(0)
    if ((re.search(r'\b'+"путин", text.lower())) or \
        (re.search(r'\b'+"медведев", text.lower()))) and \
           (re.search(r'\b'+"президент рф", text.lower())):
                meeting[i]=1                    


    i=i+1

print(sum(meeting))

dfnew1=dfnew[dfnew['2012 list']==1]

obsnum=dfnew1.groupby(["Year","Quarter", "governor"]).size()

dfnew1=dfnew1.drop(columns=['date'])
dfnew1=dfnew1.drop(columns=["clean text"])
dfnew1=dfnew1.drop(columns=["rawtext"])
#dfnew1["president"]=meeting

df3 = dfnew1.groupby(["Year","Quarter", "governor"]).sum()

print("starting part 1")

df3["Total Articles"]=obsnum

print("ending part 1")

df3.sort_values(by=['Year','Quarter'], ascending=True)

os.makedirs("C:/Users/Alex/Desktop/RegionalOctober", exist_ok=True)

print("ending part 2")

writer = pd.ExcelWriter("C:/Users/Alex/Desktop/federaldatasetOctober2023/federaltest 2012 quarter clean dup.xlsx", engine='xlsxwriter')
writer.book.use_zip64()
df3.to_excel(writer, merge_cells=False)
writer.close()

print("ending part 3")


dfnew1=dfnew[dfnew['2010 list']==1]

obsnum=dfnew1.groupby(["Year","Quarter", "governor"]).size()

dfnew1=dfnew1.drop(columns=['date'])
dfnew1=dfnew1.drop(columns=["clean text"])
dfnew1=dfnew1.drop(columns=["rawtext"])
#dfnew1["president"]=meeting

df3 = dfnew1.groupby(["Year","Quarter", "governor"]).sum()

print("starting part 1")

df3["Total Articles"]=obsnum

print("ending part 1")

df3.sort_values(by=['Year','Quarter'], ascending=True)

os.makedirs("C:/Users/Alex/Desktop/RegionalOctober", exist_ok=True)

print("ending part 2")

writer = pd.ExcelWriter("C:/Users/Alex/Desktop/federaldatasetOctober2023/federaltest 2010 quarter clean dup.xlsx", engine='xlsxwriter')
writer.book.use_zip64()
df3.to_excel(writer, merge_cells=False)
writer.close()

print("ending part 3")


"""
"""


#5. Aggregate by governor, months and sources
"""

"""
obsnum=dfnew.groupby(["Year","Month", "governor"]).size()

meeting=[]
i=0

for text in cleansentences:
    meeting.append(0)
    if ((re.search(r'\b'+"путин", text.lower())) or \
        (re.search(r'\b'+"медведев", text.lower()))) and \
           (re.search(r'\b'+"президент рф", text.lower())):
                meeting[i]=1                    


    i=i+1

print(sum(meeting))



obsnum=dfnew.groupby(["Year","Month", "governor"]).size()

dfnew=dfnew.drop(columns=['date'])
dfnew=dfnew.drop(columns=["clean text"])
dfnew=dfnew.drop(columns=["rawtext"])
#dfnew["president"]=meeting

df2 = dfnew.groupby(["Year","Month", "governor"]).sum()

print("starting part 1")

df2["Total Articles"]=obsnum

print("ending part 1")

df2.sort_values(by=['Year','Month'], ascending=True)

os.makedirs("C:/Users/Alex/Desktop/federaldatasetOctober2023", exist_ok=True)

print("ending part 2")

writer = pd.ExcelWriter("C:/Users/Alex/Desktop/federaldatasetOctober2023/federaltest all monthtly all sources clean dup.xlsx", engine='xlsxwriter')
writer.book.use_zip64()
df2.to_excel(writer, merge_cells=False)
writer.close()

print("ending part 3")


# Only 2010 sources

dfnew1=dfnew[dfnew['2010 list']==1]

obsnum=dfnew1.groupby(["Year","Month", "governor"]).size()
df3 = dfnew1.groupby(["Year","Month", "governor"]).sum()

df3["Total Articles"]=obsnum

import os

df3.sort_values(by=['Year','Month'], ascending=True)

os.makedirs("C:/Users/Alex/Desktop/federaldataset", exist_ok=True)

writer = pd.ExcelWriter("C:/Users/Alex/Desktop/federaldatasetOctober2023/federaltest all monthtly 2010 sources clean dup.xlsx", engine='xlsxwriter')
writer.book.use_zip64()
df3.to_excel(writer, merge_cells=False)
writer.close()

print("ending part 4")


# Only 2012 sources

dfnew1=dfnew[dfnew['2012 list']==1]

obsnum=dfnew1.groupby(["Year","Month", "governor"]).size()
df3 = dfnew1.groupby(["Year","Month", "governor"]).sum()

df3["Total Articles"]=obsnum

import os

df3.sort_values(by=['Year','Month'], ascending=True)

os.makedirs("C:/Users/Alex/Desktop/federaldataset", exist_ok=True)

writer = pd.ExcelWriter("C:/Users/Alex/Desktop/federaldatasetOctober2023/federaltest all monthtly 2012 sources clean dup.xlsx", engine='xlsxwriter')
writer.book.use_zip64()
df3.to_excel(writer, merge_cells=False)
writer.close()

print("ending part 5")


