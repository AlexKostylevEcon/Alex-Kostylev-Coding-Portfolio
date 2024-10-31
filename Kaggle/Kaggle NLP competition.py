# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 14:43:28 2023

@author: alexs
"""

# Code for the Kaggle NLP competition, not submitted due to the time constraints.

import pandas as pd 
import matplotlib
from matplotlib import pyplot as plt
import sys
import collections
import re
from collections import Counter
import string
import time
import sys

# Feature Engineering Function

def features(arg):
  #for idnum in range(0,len(arg)):
    
    #exampleinput = input1.loc[input1.id==idset[idnum]]
    
    exampleinput = arg
    exampleinput.reset_index(inplace = True, drop= True)
    exampleinput=exampleinput.assign(Index=exampleinput.index)
    
    # Time it took to write the text 
    
    time=exampleinput['up_time'].iloc[-1]-exampleinput['down_time'].iloc[0]
    exampleinput=exampleinput.assign(time=exampleinput["up_time"]-exampleinput["down_time"])
    timenotactive=exampleinput.loc[(exampleinput["activity"]=="Nonproduction"),"time"].sum()
    timeinput=exampleinput.loc[(exampleinput["activity"]=="Input"),"time"].sum()
    
    timenothing=time-exampleinput["time"].sum()
    timesomething=exampleinput["time"].sum()
    timesomethingshare=timesomething/timenothing
    timeinputshare=timeinput/timesomething
    
    exampleinput.drop(exampleinput[exampleinput["activity"]=="Nonproduction"].index, inplace=True)
    exampleinput = exampleinput.drop(columns=['down_time', 'up_time', 'up_event', 'word_count', 'action_time', 'time'], axis=1)
    
    #exampleinput["word change"]=exampleinput["word_count"].diff()
    exampleinput=exampleinput.assign(cursor_change=exampleinput["cursor_position"].diff())
    
    exampleinput=exampleinput.assign(sorting=0)
    
    # Feature engineering for the general text
    
    # Number of corrections
    
    backcount=exampleinput.loc[exampleinput.activity == 'Remove/Cut', 'activity'].count()
    repcount=exampleinput.loc[exampleinput.activity == 'Replace', 'activity'].count()
    correctionscount=backcount+repcount

    # Number of paste 
    pastecount=exampleinput.loc[exampleinput.activity == 'Paste', 'activity'].count()

    # Number of move
    movecount=exampleinput.loc[exampleinput.activity.str.contains("Move"), 'activity'].count()
        
    # Average length of corrections
    if backcount>0:
       avcorrection=exampleinput.loc[exampleinput["activity"]=="Remove/Cut","text_change"].apply(lambda x: len(x)).sum()/backcount
    else:
       avcorrection=0

    #######################################
    # Generating string with the final text
    
    # Backspace works, input works. 
        
    for index, row in exampleinput.iterrows():        
        
        row=exampleinput.loc[exampleinput["event_id"]==row["event_id"]]
        row=row.squeeze(axis=0)

        # What to do with ascii symbols?
        # The most obvious thing would be to simply ignore and decrease cursor position 
        # by the number of ignored characters, or increase for remove/cut
    
        
        textrow=''.join(c for c in row["text_change"] if c in string.printable)
        wronglettercount=len(row["text_change"])-len(textrow)
        if wronglettercount>0:
            try:
               textrow=row["text_change"].encode('latin1').decode('utf-8')
            except:
               textrow=row["text_change"].encode('latin1').decode('cp1251')

        if row["activity"]=="Remove/Cut":
            
            #deleting rows corresponding to text deleted by backspace
            curposstart=row["cursor_position"]+1
            relind=exampleinput["Index"][(exampleinput["cursor_position"]>=row["cursor_position"]+1) & (exampleinput["cursor_position"]<=row["cursor_position"]+len(textrow)) & (exampleinput["activity"]=="Input") & (exampleinput["Index"] < row["Index"])].iloc[-1]
            exampleinput=exampleinput.drop(index=relind)                
                        
            #changing cursor position for inputs after the backspace
            exampleinput.loc[(exampleinput["cursor_position"]>row["cursor_position"])  & (exampleinput["Index"]<row["Index"]), "cursor_position"]= exampleinput["cursor_position"]-len(textrow)                
            exampleinput=exampleinput.drop(index=row["Index"])
                
        if row["activity"]=="Input":
            exampleinput.loc[(exampleinput["cursor_position"]>row["cursor_position"]-len(textrow))  & (exampleinput["Index"]<row["Index"]), "cursor_position"]= exampleinput["cursor_position"]+len(textrow)                

        if "Move" in row["activity"]:
                        
            oldstart=int(re.findall(r'\d+', row["activity"])[0])
            oldend=int(re.findall(r'\d+', row["activity"])[1])
            newstart=int(re.findall(r'\d+', row["activity"])[2])
            newend=int(re.findall(r'\d+', row["activity"])[3])
            movedtext=textrow
            lenmove=newend-newstart
            
            diff=oldstart-newstart
                
            relrows=exampleinput.loc[(exampleinput["cursor_position"]>oldstart) & (exampleinput["cursor_position"]<=oldend) & (exampleinput["Index"]<row["Index"])]

            indlist=[]
            oldlist=[]
            
            indlist=exampleinput.loc[(exampleinput["cursor_position"]>=oldstart+1) & (exampleinput["cursor_position"]<=oldend) & (exampleinput["activity"]=="Input") & (exampleinput["Index"] < row["Index"]),"Index"]
            curlist=exampleinput.loc[(exampleinput["cursor_position"]>=oldstart+1) & (exampleinput["cursor_position"]<=oldend) & (exampleinput["activity"]=="Input") & (exampleinput["Index"] < row["Index"]),"cursor_position"]
            oldcurpos=[x-diff for x in curlist]
            oldlist=exampleinput.loc[(exampleinput["cursor_position"].isin(oldcurpos)) & (exampleinput["activity"]=="Input") & (exampleinput["Index"] < row["Index"]), "Index"]  
            
            exampleinput.loc[(exampleinput["Index"].isin(indlist)), "cursor_position"]= exampleinput["cursor_position"]-diff           
    
            exampleinput.loc[(exampleinput["Index"].isin(oldlist)) & (~exampleinput["Index"].isin(indlist)) & (oldstart<newstart),"cursor_position"]=exampleinput["cursor_position"]-lenmove
            exampleinput.loc[(exampleinput["Index"].isin(oldlist)) & (~exampleinput["Index"].isin(indlist)) & (oldstart>newstart),"cursor_position"]=exampleinput["cursor_position"]+lenmove

            exampleinput.loc[(oldstart<newstart) & (~exampleinput["Index"].isin(oldlist)) & (~exampleinput["Index"].isin(indlist)) & (exampleinput["cursor_position"]>oldend) & (exampleinput["cursor_position"]<=newstart), "cursor_position"]= exampleinput["cursor_position"]-lenmove
            exampleinput.loc[(oldstart>newstart) & (~exampleinput["Index"].isin(oldlist)) & (~exampleinput["Index"].isin(indlist)) & (exampleinput["cursor_position"]>newend) & (exampleinput["cursor_position"]<=oldstart), "cursor_position"]= exampleinput["cursor_position"]+lenmove
          
            exampleinput=exampleinput.drop(index=row["Index"])        
     
        
        "Replace solved"
        if row["activity"]=="Replace":  

           # replace takes cursor position of the last character in replacement line. 
           # start of replace is replace cursor position - len(replaced text)
           # need to delete characters from start of replace to start of replace + len(replaced text)
           # need to change position of all further elements by len(replaced text)-len(replacement text) (subtract cursor_position)
               
           replacementtext=textrow.partition("> ")[2]
           replacedtext=textrow.partition(" =")[0]
           oldstart=row["cursor_position"]-len(replacementtext)+1
           oldend=oldstart+len(replacedtext)
             
           for n in range(oldstart, oldend):
                relind=exampleinput["Index"][(exampleinput["cursor_position"]==n) & (exampleinput["activity"]=="Input") & (exampleinput["Index"] < row["Index"])].iloc[-1]
                exampleinput=exampleinput.drop(index=relind)
           #for i in range(0,row["Index"]):
           
           exampleinput.loc[(exampleinput['cursor_position']>(row["cursor_position"]-len(replacementtext)+len(replacedtext))) & (exampleinput["Index"]<row["Index"]), "cursor_position"]=exampleinput["cursor_position"]-(len(replacedtext)-len(replacementtext))   
             
            # The idea is to add new rows that would be uniqly identifiable later. Ideally by its relindex.
            # Could be also by cursor position 
            # Could simply create a sorting column and sort by index and sorting column
            # Then, reindex and recreate Index column
            # This is the best approach 
           
           exampleinput=exampleinput.drop(index=row["Index"])
           
           for l in range(len(replacementtext)):

               newrow={"id": 0,"event_id":0,"activity": "Input","down_event":replacementtext[l],"text_change":replacementtext[l],"cursor_position":row["cursor_position"]-len(replacementtext)+l+1,"Index": row["Index"], "cursor_change":0, "sorting":l+1}
               line = pd.DataFrame(newrow, index=[row["Index"]])
               exampleinput = pd.concat([exampleinput.iloc[:row["Index"]-1], line, exampleinput.iloc[row["Index"]-1:]])
           exampleinput=exampleinput.sort_values(['Index', 'sorting'], ascending=[True, True])
              
           exampleinput.reset_index(inplace = True, drop= True)
           exampleinput["Index"]=exampleinput.index               
               #exampleinput.loc[row["Index"]-len(replacementtext)+l+1]=newrow
            
           
           #exampleinput=exampleinput.drop(index=row["Index"])

           #exampleinput["text_change"][row["Index"]]= replacementtext
         
        if row["activity"]=="Paste":
            exampleinput.loc[(exampleinput['cursor_position']>(row["cursor_position"]-len(textrow))) & (exampleinput["Index"]<row["Index"]), "cursor_position"]=exampleinput["cursor_position"]+len(textrow)
            #exampleinput.update(exampleinput["cursor_position"].mask((exampleinput['cursor_position']>(row["cursor_position"]-len(textrow))) & (exampleinput["Index"]<row["Index"]), lambda x: x+len(textrow)))
  
            exampleinput=exampleinput.drop(index=row["Index"])

            for l in range(len(textrow)):
                newrow={"id": 0, "event_id":0,"activity": "Input","down_event":textrow[l],"text_change":textrow[l],"cursor_position":row["cursor_position"]-len(textrow)+l+1,"Index": row["Index"], "cursor_change":0, "sorting":l+1}
                line = pd.DataFrame(newrow, index=[row["Index"]])
                exampleinput = pd.concat([exampleinput.iloc[:row["Index"]-1], line, exampleinput.iloc[row["Index"]-1:]])
           
            exampleinput=exampleinput.sort_values(['Index', 'sorting'], ascending=[True, True])
            exampleinput.reset_index(inplace = True, drop= True)
            
            exampleinput=exampleinput.assign(Index=exampleinput.index )
                            
            #exampleinput=exampleinput.drop(index=row["Index"])
            
            
    # Final text as string
    
    
    exampleinput.drop(exampleinput[exampleinput["activity"].str.contains("Move")].index, inplace=True)
    exampleinput.drop(exampleinput[exampleinput["down_event"]=="Backspace"].index, inplace=True)
    
    exampleinput=exampleinput.sort_values("cursor_position")
    textlist=exampleinput["text_change"].tolist()
    finaltext=''.join(textlist)
               
    # Feature engineering for the final text
           
    # Punctuation signs
    from string import punctuation
    commacount=len(re.findall(",", finaltext))
    exclcount=len(re.findall(r'\!', finaltext))
    questcount=len(re.findall(r'\?', finaltext))
    percount=len(re.findall(r'\.', finaltext))
    braccount=len(re.findall(r'\(', finaltext))+len(re.findall(r'\)', finaltext))
    dashcount=len(re.findall(r'\-', finaltext))
    
    punctcount=commacount+braccount+dashcount
    
    # Number of sentences
    sencount=exclcount+questcount+percount
    
    # Punctuation per sentence
    punctpersent=punctcount/sencount
    
    # Dummy for "complicated" punctuation: !,?,:
    
    complicatedpunct=exclcount+questcount
    complicateddummy=0
    if complicatedpunct>0:
       complicateddummy=1
        
    # Number of paragraphs
    paragraphcount=finaltext.count('\n')
    
    # Number of characters
    charcount=finaltext.count("q")
    
    # Removing punctuation from string
    finaltext=finaltext.translate(str.maketrans('', '', string.punctuation))
    
    # Number of words. Need to delete punctuation first 
    splitstring=finaltext.split()
    numwords = len(finaltext.split())
    
    # longest word length
    maxwordlen = max(len(word) for word in finaltext.split())
    
    # Average length of words longer than 3 symbols 
    shortword = re.compile(r'\W*\b\w{1,3}\b')
    finaltext=shortword.sub('', finaltext)
    longwordcount = len(finaltext.split())
    longwordlensum=sum(len(word) for word in finaltext.split()) / len(finaltext.split())
    shortwordnum=numwords-longwordcount
    
    # Share of words with less or equal to 3 symbols
    shortwordshare=shortwordnum/numwords
    
    # Average word number per sentence
    senlen= numwords/sencount
    senlenlong= longwordcount/sencount
    
    
    #######################################
    # Adding all features into feature frame
    
    # backcount repcount correctionscount pastecount movecount avcorrection
    # time timenotactive timeinput
    # commacount exclcount questcount percount braccount dashcount punctcount
    # sencount punctpersent paragraphcount charcount numwords maxwordlen
    # longwordlensum longwordcount shortwordnum shortwordshare senlen senlenlong
    # timeinputshare timesomethingshare timesomething timenothing
    
    featdict={"id": row["id"],"backcount":backcount, "repcount":repcount, "correctionscount":correctionscount, "pastecount":pastecount,
              "movecount":movecount, "avcorrection":avcorrection, "time":time, "timenotactive":timenotactive,
              "timeinput":timeinput, "commacount":commacount, "exclcount":exclcount,
              "questcount":questcount,"percount":percount, "braccount":braccount, "dashcount":dashcount,
              "punctcount":punctcount, "sencount":sencount, "punctpersent":punctpersent,
              "paragraphcount":paragraphcount, "charcount": charcount, "numwords":numwords,
              "maxwordlen":maxwordlen, "longwordlensum":longwordlensum, "longwordcount":longwordcount,
              "shortwordnum":shortwordnum, "shortwordshare":shortwordshare, "senlen":senlen,
              "senlenlong":senlenlong, "timeinputshare":timeinputshare, "timesomethingshare":timesomethingshare,
              "timesomething":timesomething, "timenothing":timenothing}
    
    return featdict
    #return dictlist

if __name__ == '__main__':
    
    from multiprocess import Pool
        
    input1=pd.read_csv("C:/Users/alexs/linking-writing-processes-to-writing-quality/train_logs.csv")
    scores=pd.read_csv("C:/Users/alexs/linking-writing-processes-to-writing-quality/train_scores.csv")
    
    input1=input1.sort_values(by='id', ascending=False)
    scores=scores.sort_values(by='id', ascending=False)

    idset=list(set(input1["id"]))
    idset.sort()
    
    #input1.set_index(keys=['id'], drop=False,inplace=True)
    # Correcting what I believe is a mistake in the train data: event_id=1877 for id=b475c8d9
    # Replace of 4 symbols with cursor position of 0: 
    # inconsistent with how replace operations are recorded in all other observations 
    
    input1.loc[(input1["id"]=="b475c8d9") & (input1["event_id"]==1877),"activity"]="Input"
    input1.loc[(input1["id"]=="b475c8d9") & (input1["event_id"]==1877),"text_change"]='\n'
    input1.loc[(input1["id"]=="b475c8d9") & (input1["event_id"]==1877),"cursor_position"]=1384
        
        
    import time
    starttime=time.time()
    pool = Pool(4)
    
    #frameslist=[]
    
    #for ids in idset:
    #    frameslist.append(input1.loc[input1.id==ids])
    
    ans = [y for x, y in input1.groupby('id')]
        
    result = pool.map(features, ans)
    pool.close()
    pool.join()
    featureframe=pd.DataFrame(result)
    
    end_time=time.time()-starttime
    print(end_time)
        
    from sklearn.preprocessing import StandardScaler
    from sklearn.datasets import make_classification
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import cross_val_score
    from sklearn.model_selection import KFold
    from numpy import mean
    from numpy import std
    
    # Feature Scaling
    #scaler = StandardScaler()
    #BagFeaturesframe = scaler.fit_transform(BagFeaturesframe)
    
    featureframe=featureframe.drop(columns=["id"])
    scores=scores.drop(columns=["id"])

    
    # I want a score of 0.58 at least
    # Currently with test_size=0.1 have 0.5869

    X_train, X_test, y_train, y_test = train_test_split(featureframe, scores, test_size=0.2, random_state=0)
    
    base_model = RandomForestRegressor(n_estimators=1000, random_state=0, n_jobs=-1)
    base_model.fit(X_train, y_train) 
    y_pred = base_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**.5
    print(mse)
    print(rmse)
    
    cv = KFold(n_splits=5, random_state=1, shuffle=True)
    # evaluate model
    scor = cross_val_score(base_model, featureframe, scores, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
    # report performance
    print('f1: %.3f (%.3f)' % (mean(scor), std(scor)))
    
    
    models_dict = {}
    scor = []
    
    test_predict_list = []
    best_params = {'boosting_type': 'gbdt', 
            'metric': 'rmse',
            'reg_alpha': 0.003188447814669599, 
            'reg_lambda': 0.0010228604507564066, 
            'colsample_bytree': 0.5420247656839267, 
            'subsample': 0.9778252382803456, 
            'feature_fraction': 0.8,
            'bagging_freq': 1,
            'bagging_fraction': 0.75,
            'learning_rate': 0.01716485155812008, 
            'num_leaves': 19, 
            'min_child_samples': 46,
            'verbosity': -1,
            'random_state': 42,
            'n_estimators': 500,
             'device_type':'gpu'}
    
    # from gensim.models import Word2Vec
    
    from sklearn.metrics import mean_squared_error
    from sklearn.model_selection import KFold
    from sklearn.model_selection import cross_val_score
    from numpy import mean
    from numpy import std
    
    import lightgbm as lgb
    from sklearn import metrics, model_selection, preprocessing, linear_model, ensemble, decomposition, tree

    params = {
        "objective": "regression",
        "metric": "rmse",
        'random_state': 42,
        "n_estimators" : 12001,
        "verbosity": -1,
        "device_type" : "gpu",
        **best_params
    }

    model = lgb.LGBMRegressor(**params)

    model.fit(X_train, y_train) 
    y_pred = base_model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = mse**.5
    print(mse)
    print(rmse) 

    cv = KFold(n_splits=10, random_state=1, shuffle=True)
    # evaluate model
    scor = cross_val_score(model, featureframe, scores, scoring='neg_root_mean_squared_error', cv=cv, n_jobs=-1)
    # report performance
    print('f1: %.3f (%.3f)' % (mean(scor), std(scor)))
        
# Some replacement text contains unrecognised character followed by two spaces. These spaces are not counted in coursor position and are likely an error

