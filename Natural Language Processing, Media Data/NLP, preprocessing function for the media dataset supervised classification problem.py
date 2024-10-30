# -*- coding: utf-8 -*-
"""
Created on Wed Oct 30 11:51:55 2024

@author: Alex

This project is the part of the data generation and supervised ML classification for the main chapter in my PhD thesis.
The input is a dataset of the Russian federal media articles mentioning governors between 2002 and 2008.
The output is a binary classification into "Article mentioning governor engaging with the federal center".

The code can be separated into: 
    1. Preprocessing (inc. text data cleaning, feature engineering, sensitivity analysis)
    2. Supervised ML training
    3. Supervised ML model deployment on the whole dataset.
"""

# First, import packages

import nltk
import pandas as pd
from nltk import sent_tokenize
import re
import numpy as np
nltk.download("stopwords")
from nltk.corpus import stopwords
russian_stopwords = stopwords.words("russian")
import ru_core_news_lg
import time
nlp=ru_core_news_lg.load()



# Define pre-processing function

def preprocessing(texts, governors, regional_capital, ru_capitals, foreign_cities_dict, midfeddict, foreign_countries_dict, representative_dictionary, federal_dictionary, dates):

    spacyinput=[]
    textinput1=[]
    searchwordslist=[]
    
    # Creating the list of all governors in sample and stemming it for in-text recognition
    
    from nltk.stem.snowball import SnowballStemmer 
    stemmer = SnowballStemmer("russian")
    searchwordslist=[stemmer.stem(x) for x in governors]
        
    # Creating a list of relevant texts that include the title and name of any governor in sample   
        
    for index in range(len(texts)):
          relevant2=[]
          sentences=[]
          searchwords=searchwordslist[index]
          
          if (searchwords.lower() in texts[index].lower()):
                  textinput1.append(texts[index])
          else: 
                textinput1.append("")
    
    # Omitting texts that reference governors' full namesakes 
    
    textinput=[]
    textprob=[]
    for index in range(len(textinput1)):
          
          if (re.search(r'\b'+"итар-тасс"+r'(\w*\W+){0,8}'+"зеленин", textinput1[index].lower())) or \
              (re.search(r'\b'+"новый регион"+r'(\w*\W+){0,8}'+"волков", textinput1[index].lower())) or \
                 (re.search(r'\b'+"актер"+r'(\w*\W+){0,8}'+"морозов", textinput1[index].lower())):
              print("yes, Zelenin")
              textprob.append(textinput1[index])
              textinput.append("")
          else: 
              textinput.append(textinput1[index])  
    
    # Bringing media articles to a common format
    # Trimming meta data

    n=0
    for document in textinput:
        try:
           document=document.split("Заглавие:",1)[1] 
        except:
            n=n+1
        document=re.sub('ё', 'е', document)
        document = re.sub(r'\s+', ' ', document, flags=re.I)
        spacyinput.append(document)
        
    print("Number of problems with ЗАГЛАВИЕ:", n)   
    
    # Identifying sentences that mention governor and adjacent sentences
      
    relevantsentences=[]
    searchwordslist=[]
    sentlength=[]  
    problemgovernor=[]
    problemindex=[]
    problemtext=[]
    problemclean=[]
        
    for index in range(len(spacyinput)):
      relevant2=[]
      sentences=[]
      searchwords=searchwordslist[index]
      sentences = sent_tokenize(spacyinput[index])
          
      for sentence in sentences:
          
          relindex=sentences.index(sentence)
          
          if (searchwords.lower() in sentence.lower()):
             
              sentence=re.sub(r'\w*'+searchwords+'\w*', 'searchword', sentence,flags=re.IGNORECASE)
              sentences[relindex]=sentence

              if relindex<(len(sentences)-1) and relindex>0:
                     relevant2.append(sentences[relindex-1])
                     relevant2.append(sentences[relindex])
                     relevant2.append(sentences[relindex+1])
              else:
                  if relindex>0:
                      relevant2.append(sentences[relindex-1])
                      relevant2.append(sentences[relindex])
                  else:
                        if (len(sentences)-1)>0:
                          relevant2.append(sentences[relindex])
                          relevant2.append(sentences[relindex+1])
                        else: 
                          relevant2.append(sentences[relindex])
      
      # Creating a list of texts where the governor mention was not identified for manual check  
      if relevant2==[]:
          problemgovernor.append(searchwords)
          problemindex.append(index)
          problemtext.append(spacyinput[index])
          problemclean.append(sentences)
      
      # Keeping only unique sentences and creating a list with relevant sentences
      
      unique=list(set(relevant2))
      sentlength.append(len(unique))
      relevantsentence=' '.join(unique)      
      relevantsentences.append(relevantsentence)
      print(len(relevantsentences))
    
    # Extracting first set of features from texts: dates, sentence length, text sentiment
    # Need to change this. Make this self-sufficient
    
    datelist=[]
    rawtexts=[]
    reg_capital=[]
    governorall=[]
    sentlen=[]
    
    # Sentiment Analysis
    
    from dostoevsky.tokenization import RegexTokenizer
    from dostoevsky.models import FastTextSocialNetworkModel
    
    tokenizer = RegexTokenizer()
    model = FastTextSocialNetworkModel(tokenizer=tokenizer)
    results = model.predict(relevantsentences)
    
    negatives=[]
    positives=[]
    neutrals=[]
    
    for x in results:
        negatives.append(x["negative"])
        positives.append(x["positive"])
        neutrals.append(x["neutral"])
    
    negatives=[0.3 if x>0.3 else x for x in negatives]    
    positives=[0.3 if x>0.3 else x for x in positives]    
    
    for p in range(len(texts)):
       if relevantsentences[p]!="":
           reg_capital.append(regional_capital[p])
           rawtexts.append(texts[p])
           governorall.append(governors[p])
           datelist.append(dates[p])
    
    relevantsentences = list(filter(None, relevantsentences))
        
    # Generating punctuation, entities and text length features
        
    excllist=[]
    commalist=[]
    
    for sentence in relevantsentences:
        exclcount=0
        commacount=0
        for word in sentence:
            if "!" in word:
                exclcount=exclcount+1
            if "," in word:
                commacount=commacount+1
        excllist.append(exclcount)
        commalist.append(commacount)
       
    import ru_core_news_lg
    nlp=ru_core_news_lg.load()
    
    entitieslist=[]
    entitliesyeslist=[]
    entitiespers=[]
    entitiespersyes=[]
    entitiesloc=[]
    entitieslocyes=[]
    entorglist=[]
    entorgyeslist=[]
    
    l=-1    
    for document in nlp.pipe(relevantsentences):
        l=l+1
        #document=nlp(sentence)
        entities=0
        entitiesyes=0
        entloc=0
        entlocyes=0
        entper=0
        entperyes=0
        entorg=0
        entorgyes=0
        for token in document:
            if token.ent_iob_=="I":
               entities=entities+1
               if token.ent_type_=="LOC":
                   entloc=entloc+1
                   print("Location: ", token)
               if token.ent_type_=="PER":
                   entper=entper+1
                   print("Person:", token)
               if token.ent_type_=="ORG":
                   #entorg=entorg+1
                   print("Enterprise:", token)
            if entities>0:
                entitiesyes=1
            if   entloc>0:
                entlocyes=1
            if   entper>0:
                entperyes=1 
            if   entorg>0:
                 entorgyes=1     
        entitieslist.append(entities/sentlen[l])   
        entitliesyeslist.append(entitiesyes)    
        entitiespers.append(entper/sentlen[l])
        entitiespersyes.append(entperyes)
        entitiesloc.append(entloc/sentlen[l])
        entitieslocyes.append(entlocyes)
        entorglist.append(entorg/sentlen[l])
        entorgyeslist.append(entorgyes)
        
    # Final cleaning and text lemmitisation 
      
    cleaninput=[]
     
    for sent in relevantsentences: 
          sentence = sent
          sentence = sentence.lower()
          sentence = ' '.join([word for word in sentence.split() if word not in russian_stopwords])
          sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)
          sentence = re.sub(r'[^\w\s\d\.]+', ' ', str(sentence)) 
          sentence = re.sub(r'[^\w\.]\s+', ' ', str(sentence)) 
          # remove all single characters
          sentence = re.sub(r'\s+[a-zA-Z]\s+', ' ', sentence)
          # Remove single characters from the start
          sentence = re.sub(r'\^[a-zA-Z]\s+', ' ', sentence) 
          # Substituting multiple spaces with single space
          sentence = re.sub(r'\s+', ' ', sentence, flags=re.I)
          # Removing prefixed 'b'
          sentence = re.sub(r'^b\s+', '', sentence)
          # Converting to Lowercase
          sentence = sentence.lower()
          # Deleting stopwords
          sentence = ' '.join([word for word in sentence.split() if word not in russian_stopwords])
          # Deleting numbers
          sentence=re.sub(r'[0-9]+', '', sentence)

          cleaninput.append(sentence)   
                 
    # Lemmatizing with spacy.
      
    cleansentences=[]
    
    print("Started")    
    for document in nlp.pipe(cleaninput):
        document = [token.lemma_.lower() for token in document] 
        #if token.ent_type_!="PER" else token.lower()
        document = ' '.join(document)
        cleansentences.append(document)
    print("Done") 

    ## Feature Engineering 
    
    # Federal Officials inquiry 1
    # F1 0.42. Kind of useful, but not on its own. Need high precision. 
    
    federalinq=[]
    i=0
    
    for sentence in cleansentences:
        federalinq.append(0)
        sents = sent_tokenize(sentence)
        for s in sents:
            
            if ((re.search(r'\b'+"путин", s.lower())) and \
                (re.search(r'\b'+"searchword", s.lower()))) or \
                ((re.search(r'\b'+"медведев", s.lower())) and \
                    (re.search(r'\b'+"searchword", s.lower()))) or \
                    ((re.search(r'\b'+"президент рф", s.lower())) and \
                        (re.search(r'\b'+"searchword", s.lower()))) or \
                        ((re.search(r'\b'+"глава государст", s.lower())) and \
                            (re.search(r'\b'+"searchword", s.lower()))):
                    federalinq[i]=1
        
        i=i+1

    print(sum(federalinq))
    
    # Checking inquiry performance
    # Very good precision, 0.87, if appointed are included in fedearl officials. f1 0.478
    # With no appointment/firing news precision is very high, 0.91. f1 0.488
    
    # Federal inquiry 2
    # Prime ministers of Russia by surname
    
    pmlist=["касьянов", "христенко", "фрадков", "зубков"]
    
    federalinq2=[]
    i=0
    
    for sentence in cleansentences:
        federalinq2.append(0)
        sents = sent_tokenize(sentence)
        for s in sents:
            
            for name in pmlist:
               
              if ((re.search(r'\b'+name, s.lower())) and \
                  (re.search(r'\b'+"searchword", s.lower()))):
                      federalinq2[i]=1  
        i=i+1
    
    print(sum(federalinq2))
    
    #Inquiry 3
    # Federation council
    # Not a strong predictor
    
    federalinq3=[]
    i=0
    
    for sentence in cleansentences:
        federalinq3.append(0)
        sents = sent_tokenize(sentence)
        for s in sents:
            
              if ((re.search(r'\b'+"совет федерац", s.lower())) and \
                  (re.search(r'\b'+"searchword", s.lower()))):
                      federalinq2[i]=1   
        i=i+1
    
    print(sum(federalinq3))
    
    #Inquiry 3
    # State council works well enough, precision of 0.84. Does not have high correlation with the president mentions.
    
    federalinq3=[]
    i=0
    
    for sentence in cleansentences:
        federalinq3.append(0)
        
        if (re.search(r'\b'+"совет рф", s.lower())) or \
            (re.search(r'\b'+"госсовет российский", s.lower())) or \
            (re.search(r'\b'+"госсовет рф", s.lower())) or \
                (re.search(r'\b'+"совет российский", s.lower())):
                federalinq3[i]=1     
        i=i+1
    
    print(sum(federalinq3))
         
    # Inquiry 4, polpreds - surnames 
    # Surprisingly, the name list does not work well. Low precision of 0.5
    polpredlist=["говорун", "черкесов", \
            "клебанов", "винниченко", "казанцев", "козак", "рапота", "устинов",\
             "кириенко", "коновалов", "бабич", "латышев", "куйвашев", \
                    "драчевский", "квашнин", "пуликовский", "исхаков", "сафонов"]
    
    federalinq4=[]
    i=0
    
    for sentence in cleansentences:
        federalinq4.append(0)
        sents = sent_tokenize(sentence)
        for s in sents:
            
            for name in polpredlist:
            
                if ((re.search(r'\b'+name, s.lower())) and \
                    (re.search(r'\b'+"searchword", s.lower()))):
                        federalinq4[i]=1               
        i=i+1
    
    print(sum(federalinq4))
    
    # Inquiry 4, Polpreds - only titles. 
    # The title inquiry works better.
    
    federalinq5=[]
    i=0
    
    for sentence in cleansentences:
        federalinq5.append(0)
        sents = sent_tokenize(sentence)
            
        if ((re.search(r'\b'+"полпред", s.lower())) and \
            (re.search(r'\b'+"searchword", s.lower()))) or \
            ((re.search(r'\b'+"полномочный представитель президент", s.lower())) and \
                (re.search(r'\b'+"searchword", s.lower()))):
                 federalinq5[i]=1
        i=i+1
    
    print(sum(federalinq5))
    
    # Inquiry 6, Ministers, UR, Parliament

    federalinq6=[]
    i=0
    
    for sentence in cleansentences:
        federalinq6.append(0)
        sents = sent_tokenize(sentence)
        for s in sents:

              if ((re.search(r'\b'+"министр", s.lower())) and \
                  (re.search(r'\b'+"рф", s.lower())) and \
                  (re.search(r'\b'+"searchword", s.lower()))):
                      federalinq6[i]=1       
        i=i+1
    
    print(sum(federalinq6))
    
    # Federal inquiry 7, Political parties
    
    federalinq7=[]
    i=0
    
    for sentence in cleansentences:
        federalinq7.append(0)
        sents = sent_tokenize(sentence)
        for s in sents:
  
              if ((re.search(r'\b'+"единый россия", s.lower())) and \
                  not(re.search(r'\b'+"кпрф", s.lower())) and \
                      not(re.search(r'\b'+"лдпр", s.lower())) and \
                  (re.search(r'\b'+"searchword", s.lower()))):
                      federalinq7[i]=1
        i=i+1
    
    print(sum(federalinq7))
    
    # Federal inquiry 7, Grizlov, Mironov 
    
    federalinq8=[]
    i=0
    
    for sentence in cleansentences:
        federalinq8.append(0)
        sents = sent_tokenize(sentence)
        for s in sents:
            if ((re.search(r'\b'+"грызлов", s.lower())) or \
                (re.search(r'\b'+"миронов", s.lower()))) and \
                   (re.search(r'\b'+"searchword", s.lower())):
                        federalinq8[i]=1                    
        i=i+1
    
    print(sum(federalinq8))
    
    # Parts of speach features
    
    nounlist=[]
    verblist=[]
    adjlist=[]
    advlist=[]
    detlist=[]
    cconjlist=[]
    adplist=[]
    for cleansent in cleansentences:
        document=nlp(cleansent)
        noun=0
        adv=0
        adj=0
        verb=0
        adp=0
        det=0
        cconj=0
        length=len(cleansent.split())
        for token in document:
            if token.pos_=="NOUN":
                noun=noun+1
            if token.pos_=="ADV":
                adv=adv+1
            if token.pos_=="ADP":
                adp=adp+1
            if token.pos_=="ADJ":
                adj=adj+1
            if token.pos_=="VERB":
                verb=verb+1
            if token.pos_=="DET":
                det=det+1
            if token.pos_=="CCONJ":
                cconj=cconj+1    
        nounlist.append(noun/length)
        advlist.append(adv/length)
        adjlist.append(adj/length)
        adplist.append(adp/length)
        verblist.append(verb/length)
        detlist.append(det/length)
        cconjlist.append(cconj/length) 
    
    ## Dictionaries-based features
        
    dictwordslist=[]
    numwordslist=[]
    dictsharelist=[]
    dictyeslist=[]
    
    feddictsharelist=[]
    feddictyeslist=[]
    feddictwordslist=[]
    
    global econdict
    global socialdict
    
    econdict=[x for x in econdict if x==x]
    socialdict=[x for x in socialdict if x==x]
    
    # Lemmatising dictionaries
    
    cleanEcondict=[]
    for x in range(len(econdict)):
        document=nlp(econdict[x])
        document = [token.lemma_.lower() for token in document]
        document = ' '.join(document)
        cleanEcondict.append(document)
        
    cleanSocialdict=[]
    for x in range(len(socialdict)):
        document=nlp(socialdict[x])
        document = [token.lemma_.lower() for token in document]
        document = ' '.join(document)
        cleanSocialdict.append(document)    
    
    cleanRepresentativedict=[]
    for x in range(len(representative_dictionary)):
        document=nlp(representative_dictionary[x])
        document = [token.lemma_.lower() for token in document]
        document = ' '.join(document)
        cleanRepresentativedict.append(document)
        
    cleanFeddict=[]
    for x in range(len(federal_dictionary)):
        document=nlp(federal_dictionary[x])
        document = [token.lemma_.lower() for token in document]
        document = ' '.join(document)
        cleanFeddict.append(document)    
    
    cleanforeigncities=[]
    for x in range(len(foreign_cities_dict)):
        document=nlp(foreign_cities_dict[x])
        document = [token.lemma_.lower() for token in document]
        document = ' '.join(document)
        cleanforeigncities.append(document)
    
    cleanforeigncountries=[]
    for x in range(len(foreign_countries_dict)):
        document=nlp(foreign_countries_dict[x])
        document = [token.lemma_.lower() for token in document]
        document = ' '.join(document)
        cleanforeigncountries.append(document)
    
    cleanregcaps=[]
    for x in range(len(ru_capitals)):  
        document=nlp(ru_capitals[x])
        document = [token.lemma_.lower() for token in document]
        document = ' '.join(document)
        cleanregcaps.append(document)    
    
    # Match with the lists of foreign cities and countries and regional capitals
    
    foreign_cities_list=[]
    foreign_countr_list=[]
    foreign_cities_yeslist=[]
    foreign_countr_yeslist=[]
    ru_cities_list=[]
    ru_cities_yeslist=[]   
    dictwordslist=[] 
    numwordslist=[]
    governornumlist=[]
    governorsharelist=[]
    governoryeslist=[]
    
    midfedlist=[]
    
    econdictsharelist=[]
    socialdictsharelist=[]
    
    m=0
    
    for k in range(len(cleansentences)):
        
        foreign_cities_words=0
        foreign_countr_words=0
        ru_cities_words=0
        
        midfedword=0
        for word in midfeddict:
            if re.search(r'\b' + word + r'\b',  cleansentences[k]):
           # if word in cleansentences[k]:
                midfedword=midfedword+1
        midfedlist.append(midfedword)
        
        ru_caps_words=sum([char in cleansentences[k] for char in cleanregcaps if cleanregcaps!=reg_capital[k]])
        
        for word in cleanregcaps:
            if word!=reg_capital[k].lower() and word!="владимир" and governorall[k]!="Кузнецов":
               if re.search(r'\b' + word + r'\b',  cleansentences[k]):
                  ru_cities_words=ru_cities_words+1
        ru_cities_list.append(ru_cities_words)
        
        for word in cleanforeigncities:
            if re.search(r'\b' + word + r'\b',  cleansentences[k]):
                foreign_cities_words=foreign_cities_words+1
        foreign_cities_list.append(foreign_cities_words)
        
        for word in cleanforeigncountries:
            if re.search(r'\b' + word + r'\b',  cleansentences[k]):
                foreign_countr_words=foreign_countr_words+1
                print(word)
        foreign_countr_list.append(foreign_countr_words)
    
        econdictwords=0
        
        for word in cleanEcondict:
            if re.search(r'\b' + word + r'\b',  cleansentences[k]):
                econdictwords=econdictwords+1
                print(word)
                
        socialdictwords=0
        
        for word in cleanSocialdict:
            if re.search(r'\b' + word + r'\b',  cleansentences[k]):
                socialdictwords=socialdictwords+1
                print(word)
             
        for word in ru_cities_list:
         ru_cities_yes=0   
         if word>=1:
           ru_cities_yes=1
        ru_cities_yeslist.append(ru_cities_yes)
    
        for word in foreign_cities_list:
         foreign_cities_yes=0
         if word>=1:
           foreign_cities_yes=1
        foreign_cities_yeslist.append(foreign_cities_yes)
        
        for word in foreign_countr_list:
         foreign_countr_yes=0
         if word>=1:
           foreign_countr_yes=1
        foreign_countr_yeslist.append(foreign_countr_yes)
        
        numwords = len(cleansentences[k].split())
        numwordslist.append(numwords)
        
        econdictshare=econdictwords/numwords
        econdictsharelist.append(econdictshare)
        
        socialdictshare=socialdictwords/numwords
        socialdictsharelist.append(socialdictshare)
        
        # Checking how many governors are mentioned in text
        
        governornum=sum([char.lower() in cleansentences[k] for char in governorall])
        governornumlist.append(governornum)
        governorsharelist.append(governornum/numwords)
        if governornum>=1:
           governoryeslist.append(1)
        else: governoryeslist.append(0)   
    
        dictwords = sum([char.lower() in cleansentences[k] for char in cleanRepresentativedict])
        
        dictwordslist.append(dictwords)
        dictshare=dictwords/numwords
        dictsharelist.append(dictshare)
        dictyes=0
        if dictwords>=1:
            dictyes=1
        dictyeslist.append(dictyes)    
        
        feddictwords = sum([char.lower() in cleansentences[k] for char in cleanFeddict])
        
        feddictwordslist.append(feddictwords)
        feddictshare=dictwords/numwords
        feddictsharelist.append(feddictshare)
        feddictyes=0
        if feddictwords>=1:
            feddictyes=1
            print(relevantsentences[k])
        feddictyeslist.append(feddictyes)   
    
    print(sum(ru_cities_yeslist), sum(foreign_cities_yeslist), sum(foreign_countr_yeslist))
    
    # visualising the number and share of governors mentions in texts
    
    from matplotlib import pyplot as plt
    plt.figure()
    n, bins, patches = plt.hist(governorsharelist, 30, stacked=True, density = True)
    plt.show()
    plt.figure()
    n, bins, patches = plt.hist(governornumlist, 30, stacked=True, density = True)
    plt.show()
    
    commanormlist=[]
    for n in range(len(commalist)):
       commanormlist.append(commalist[n]/numwordslist[n])
    
    p1 = np.square(dictsharelist)
    p3 = np.square(numwordslist)
    p4 = np.square(dictwordslist)
    
    fedp1=np.square(feddictsharelist)
    fedp2=np.square(feddictwordslist)
        
    # Length Analysis
    
    word_count=[]
    char_count=[]
    sentence_count=[]
    avg_word_length=[]
    avg_sentence_lenght=[]
    
    for m in range(len(rawtexts)):
        word_count.append(len(str(rawtexts[m]).split(" ")))
        char_count.append(sum(len(word) for word in str(rawtexts[m]).split(" ")))
        sentence_count.append(len(str(rawtexts[m]).split(".")))
        avg_word_length.append(char_count[m] / word_count[m])
        avg_sentence_lenght.append(word_count[m] / sentence_count[m])
    
    word_count_narr=[]
    char_count_narr=[]
    avg_word_length_narr=[]
     
    for m in range(len(cleansentences)):
        word_count_narr.append(len(str(cleansentences[m]).split(" ")))
        char_count_narr.append(sum(len(word) for word in str(cleansentences[m]).split(" ")))
        avg_word_length_narr.append(char_count_narr[m] / word_count_narr[m])
        
    pword = np.square(word_count).reshape(-1, 1)
    pchar = np.square(char_count).reshape(-1, 1)
    psent = np.square(sentence_count).reshape(-1, 1)
    pavgword = np.square(avg_word_length).reshape(-1, 1)
    pavgsent = np.square(avg_sentence_lenght).reshape(-1, 1)      
    
    ## More inquiries-based features   
    # Inquiry 1. Public engagement with regional population
    
    nlpinquiry1=[]
    inquiry1=[]
    inquiry1= ["рабочий", "рабочими", "учителя", "пенсионеры", "юные", "школьники", "студенты", "дети", "врачи", "спортсмены", "красная лента", "красная ленточка", "поздравлять с праздником"]
    for word in inquiry1:
       #word=word.split()
       nlpword=nlp(word)
       inquiry1word =[x.lemma_.lower() for x in nlpword]
       nlpword=' '.join(inquiry1word)
       nlpinquiry1.append(nlpword)
    
    i=0
    engagement_inquiry=[]
    for cleansentence in cleansentences:
       engagement_inquiry.append(0) 
       s = cleansentence
       for words in nlpinquiry1:
           if re.search(r'\b'+words+r'\b', s):
          # if re.search(r'\b' + words + r'\b', s):  
            engagement_inquiry[i]=1
       i=i+1
       print("engagement with population articles: ", sum(engagement_inquiry))   
    
    # Inquiry 2. Public visits to schools, hospitals, forums, concerts etc.
    
    inquiry2=["посетил", "визит", "открыл", "осмотрел"] 
    inquiry3=["больница", "школа", "театр", "музей", "стадион", "выставка", "музыкальный","форум", "фестиваль", "концерт", "церемония"]
    
    nlpinquiry2=[]
    for word in inquiry2:
       nlpword=nlp(word)
       inquiry2word =[x.lemma_.lower() for x in nlpword]
       nlpword=' '.join(inquiry2word)
       nlpinquiry2.append(nlpword)
    
    nlpinquiry3=[]
    for word in inquiry3:
       nlpword=nlp(word)
       inquiry3word =[x.lemma_.lower() for x in nlpword]
       nlpword=' '.join(inquiry3word)
       nlpinquiry3.append(nlpword)
            
    i=0
    visit_inquiry=[]
    nlpinquirycheck2=[]
    for cleansentence in cleansentences:
       nlpinquirycheck2.append(0) 
       visit_inquiry.append(0)
       s = cleansentence
       for words in nlpinquiry2:
           if re.search(r'\b'+words+r'\b', s):
          # if re.search(r'\b' + words + r'\b', s):  
            nlpinquirycheck2[i]=1
            
            for otherwords in nlpinquiry3:
                if re.search(r'\b'+otherwords+r'\b', s):
                   visit_inquiry[i]=1
                   break
       i=i+1
       print("visits articles: ", sum(visit_inquiry))   
    
    # Inquiry 3. Speeches 
    
    inquiry4=["произносить речь", "произнес речь", "объявлять благодарность", "объявить благодарность", "объявил благодарность" "выступил речью", "губернатор речь"]
    
    speech_inquiry=[]
    nlpinquiry4=[]
    for word in inquiry4:
       nlpword=nlp(word)
       inquiry4word =[x.lemma_.lower() for x in nlpword]
       nlpword=' '.join(inquiry4word)
       nlpinquiry4.append(nlpword)
       #print(nlpinquiry4)
    
    i=0
    for sentence in cleansentences:
        speech_inquiry.append(0)
        s=sentence
        for words in nlpinquiry4:
                 words=words.split()
                 if (re.search(r'\b'+words[0]+r'(\w*\W+){0,6}'+words[1], s)) or (re.search(r'\b'+words[1]+r'(\w*\W+){0,6}'+words[0], s)):         
                    speech_inquiry[i]=1
        i=i+1
        print("speech articles: ", sum(speech_inquiry))   
        
    # inquiry 4. Festivals
    
    inquiry5=["выступил","выступление", "речь", "речи", "вступительная речь", "посетить"]
    inquiry6=["открытие", "фестиваль", "выставка","мероприятие","музыкальный","торжество", "форум", "концерт","церемония"]
    
    nlpinquiry5=[]
    for word in inquiry5:
       nlpword=nlp(word)
       inquiry5word =[x.lemma_.lower() for x in nlpword]
       nlpword=' '.join(inquiry5word)
       nlpinquiry5.append(nlpword)
    
    nlpinquiry6=[]
    for word in inquiry6:
       nlpword=nlp(word)
       inquiry6word =[x.lemma_.lower() for x in nlpword]
       nlpword=' '.join(inquiry6word)
       nlpinquiry6.append(nlpword)
            
    i=0
    festivals_inquiry=[]
    nlpinquirycheck5=[]
    for cleansentence in cleansentences:
       nlpinquirycheck5.append(0) 
       festivals_inquiry.append(0)
       s = cleansentence
       for words in nlpinquiry5:
           if re.search(r'\b'+words+r'\b', s):
          # if re.search(r'\b' + words + r'\b', s):  
            nlpinquirycheck5[i]=1
            
            for otherwords in nlpinquiry6:
                if re.search(r'\b'+otherwords+r'\b', s):
                   festivals_inquiry[i]=1
                   break
        
       i=i+1
    print("festivals articles: ", sum(festivals_inquiry))   
    
    # Inquiry 5. Award and public ceremonies
    
    inquiry7=["вручить", "наградить", "подарить","была вручена"]
    inquiry8=["медаль", "грамота","орден", "премия", "подарок"]
    
    nlpinquiry7=[]
    for word in inquiry7:
       nlpword=nlp(word)
       inquiry7word =[x.lemma_.lower() for x in nlpword]
       nlpword=' '.join(inquiry7word)
       nlpinquiry7.append(nlpword)
    
    nlpinquiry8=[]
    for word in inquiry8:
       nlpword=nlp(word)
       inquiry8word =[x.lemma_.lower() for x in nlpword]
       nlpword=' '.join(inquiry8word)
       nlpinquiry8.append(nlpword)
            
    i=0
    awards_inquiry=[]
    nlpinquirycheck7=[]
    for cleansentence in cleansentences:
       nlpinquirycheck7.append(0) 
       awards_inquiry.append(0)
       s = cleansentence
       for words in nlpinquiry7:
           if re.search(r'\b'+words+r'\b', s):
          # if re.search(r'\b' + words + r'\b', s):  
            nlpinquirycheck7[i]=1
            
            for otherwords in nlpinquiry8:
                if re.search(r'\b'+otherwords+r'\b', s):
                   awards_inquiry[i]=1
                   break    
       i=i+1
    print("awards articles: ", sum(awards_inquiry))   
    
    # Isolating political and media rating articles
    
    ratinginq=[]
    i=0
    
    for sentence in cleansentences:
        ratinginq.append(0)
        sents = sent_tokenize(sentence)
        for s in sents:
            
            if (re.search(r'\b'+"рейтинг", s.lower())):
                ratinginq[i]=1
        i=i+1
        
    for i,_ in enumerate(ratinginq):
        if ratinginq[i]==1 and governornumlist[i]>5: 
            ratinginq[i]=1
        else: 
            ratinginq[i]=0
    
    print("rating articles:", sum(ratinginq))
    
    # Isolating birthday articles
    
    birthdayinq=[]
    i=0
    
    for sentence in cleansentences:
        birthdayinq.append(0)
        sents = sent_tokenize(sentence)
        for s in sents:
            
            if (re.search(r'\b'+"день рожд", sentence.lower())):
                birthdayinq[i]=1
                print(sentence)
        i=i+1
        
    print("birthday articles: ", sum(birthdayinq))
    
    ##############################
    # Feature engineering finished, converting to Numpy arrays
    ##############################
    
    """
    Feature engineering finished
    Listing all features:
    
    Features based on "representative" public activity text inquiries
        awards_inquiry, festivals_inquiry, visit_inquiry, speech_inquiry, engagement_inquiry,
   
    Length Analysis
        pavgsent, pavgword, psent, pchar, pword, avg_word_length_narr, char_count_narr, word_count_narr,
        avg_sentence_lenght, avg_word_length, sentence_count, char_count, word_count, senlen1
    
    Dictionary Variables
        Representative activity dictionary:
            p1, p3, p4, dictyeslist, dictsharelist, dictwordslist, numwordslist, 
   
        Cities and countries variables 22
            foreign_countr_yeslist, foreign_cities_yeslist, ru_cities_yeslist,
            foreign_countr_list, foreign_cities_list, ru_cities_list
    
        Federal center dictionary and inquiries:        
            federalinq, federalinq2, federalinq3, federalinq4, federalinq5, federalinq6, federalinq7, federalinq8
            
        Other dictionaries:
            socialdictsharelist, econdictsharelist, midfedlist
            
        
    
    Part of Speech variables
        cconjlist, detlist, verblist, adplist, adjlist, advlist, nounlist
    
    Entities variables
        entorgyeslist, entorglist, entitieslocyes, entitiesloc, 
        entitiespersyes, entitiespers, entitliesyeslist, entitieslist
    
    Punctuation
        commanormlist, excllist
    
    Sensitivity variables
        positives, negatives, neutrals
    
    """
    
    # Checking correlation between features
    
    
    featureframe = pd.DataFrame({'awards_inquiry': awards_inquiry,
                                 'festivals_inquiry': festivals_inquiry,
                                 'visit_inquiry': visit_inquiry,
                                 'speech_inquiry': speech_inquiry,
                                 'engagement_inquiry': engagement_inquiry,
                                 'pavgsent': list(pavgsent.flatten()),
                                 'pavgword': list(pavgword.flatten()),
                                 'psent': list(psent.flatten()),
                                 'pchar': list(pchar.flatten()),
                                 'pword': list(pword.flatten()),
                                 'avg_word_length_narr': avg_word_length_narr,
                                 'char_count_narr': char_count_narr,
                                 'word_count_narr': word_count_narr,
                                 'avg_sentence_lenght': avg_sentence_lenght,
                                 'avg_word_length': avg_word_length,
                                 'sentence_count': sentence_count,
                                 'char_count': char_count,
                                 'word_count': word_count,
                                 'p1': list(p1.flatten()),
                                 'p3': list(p3.flatten()),
                                 'p4': list(p4.flatten()),
                                 'senlen1': sentlen,
                                 'commanormlist': commanormlist,
                                 'dictyeslist': dictyeslist,
                                 'dictsharelist': dictsharelist,
                                 'dictwordslist': dictwordslist,
                                 'numwordslist': numwordslist,
                                 'foreign_countr_yeslist': foreign_countr_yeslist,
                                 'foreign_cities_yeslist': foreign_cities_yeslist,
                                 'ru_cities_yeslist': ru_cities_yeslist,
                                 'foreign_countr_list': foreign_countr_list,
                                 'ru_cities_list': ru_cities_list,
                                 'cconjlist': cconjlist,
                                 'detlist': detlist,
                                 'verblist': verblist,
                                 'adplist': adplist,
                                 'adjlist': adjlist,
                                 'advlist': advlist,
                                 'nounlist': nounlist,
                                 'entorgyeslist': entorgyeslist,
                                 'entorglist': entorglist,
                                 'entitieslocyes': entitieslocyes,
                                 'entitiesloc': entitiesloc,
                                 'entitiespersyes': entitieslocyes,
                                 'entitiespers': entitiespers,
                                 'entitliesyeslist': entitliesyeslist,
                                 'entitieslist': entitieslist,
                                 'excllist': excllist,
                                 'feddictyeslist': feddictyeslist,
                                 'feddictsharelist': feddictsharelist,
                                 'feddictwordslist': feddictwordslist,
                                 'fedp1': fedp1, 
                                 'fedp2': fedp2,
                                 'fedearlinq1':federalinq,
                                 'fedearlinq2':federalinq2,
                                 'fedearlinq3':federalinq3,
                                 'fedearlinq4':federalinq4,
                                 'fedearlinq5':federalinq5,
                                 'fedearlinq6':federalinq6,
                                 'fedearlinq7':federalinq7,
                                 'federalinqarr8':federalinq8,
                                 'governoryeslist': governoryeslist,
                                 'governornumlist': governornumlist,
                                 'governorsharelist': governorsharelist,
                                 'birthdayinq': birthdayinq,
                                 'ratinginq':ratinginq,
                                 'positive': positives,
                                 'negative': negatives,
                                 'neutral': neutrals,
                                 "socialdictsharelist": socialdictsharelist,
                                 "econdictsharelist":econdictsharelist,
                                 "midfedlist": midfedlist
                                 })

    return cleansentences, featureframe, governorall, datelist, rawtexts