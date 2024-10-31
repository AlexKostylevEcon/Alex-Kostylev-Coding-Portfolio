# -*- coding: utf-8 -*-
"""
Created on Thu Dec 21 21:09:35 2023

@author: Alex
"""

import pandas as pd
import tiktoken
from openai import OpenAI
import os
import re

interviewstable=pd.read_excel("C:/Users/Alex/Desktop/interview inq/interview inq.xlsx")
interviews=interviewstable["rawtext"].tolist()

selectedinterviews=[interviews[0],interviews[3], interviews[5],interviews[12], interviews[13], interviews[22],interviews[36], interviews[65], interviews[67], interviews[69]]

table1 = pd.read_excel("C:/Users/Alex/Desktop/Russia. Increasing accuracy/interviews data federal part 1.xlsx")
table2 = pd.read_excel("C:/Users/Alex/Desktop/Russia. Increasing accuracy/interviews data federal part 2.xlsx")

federal=pd.concat([table1,table2])
federal.columns

federal["selected"]=0
federal.loc[(federal["text length"]>400) & (federal["person count"]<25) & (federal["other governors"]<5), "selected"]=1
federal["selected"].sum()

federal=federal.reset_index(drop=True)

federal=federal.sort_values(by=['selected','text length'], ascending=[False,True])
federal=federal.reset_index(drop=True)

print(federal["clean text"][0])
print(federal["text length"][0])
print(federal["selected"][0])

# So, the plan is to run openai model for the first 220507 pre-selected federal articles first. 
# Need to calculate the price for this.

# How reasonable is a 400 words cutoff? 400 words is a lot of sentences. Its a fairly long text

# The overall cost (not including the answer) is expected to be around 124 dollars for federal news using shortened texts and 504 using the whole texts

selected=federal["clean text"].loc[federal["selected"]==1].tolist()
selected=selected[0:100]
#selected=selectedinterviews+selected
selectedtitle=[]

for document in selected:
    try:
       document=document.split("Заглавие:",1)[1] 
    except:
        pass
    document=re.sub('ё', 'е', document)
    selectedtitle.append(document)

selected=selectedtitle

client = OpenAI()

shorttexts=[]
for text in selected:
    text=re.sub(r"(^[/,\.!?])", "", text, flags=re.MULTILINE)   
    text = os.linesep.join([s for s in text.splitlines() if s])
    sentences=text.splitlines()
    shorttexts.append("\n".join(sentences[0:16]))
    
def num_tokens_from_string(string: str, encoding_name: str) -> int:
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens

mergedinq=" ".join(selected)
shortmergedinq=" ".join(shorttexts)

selectedtitle.sample(frac=1)

tokensnumlong=num_tokens_from_string(mergedinq, "cl100k_base")
print("cost in USD: ", tokensnumlong/1000*0.0010)

tokensnumshort=num_tokens_from_string(shortmergedinq, "cl100k_base")
print("cost in USD: ", tokensnumshort/1000*0.0010)

extralonganswer=[]

#          {"role": "user", "content": 'What is the possibility that this is an interview? Give a very short answer in the first sentence. Explain in further sentences.\n Text: \"\"\"\n'+selected[i]+'\n\"\"\"'}
#          {"role": "user", "content": 'What is the possibility that this is an interview? Answer in less than 10 words in the first sentence. Explain in further sentences.\n Text: \"\"\"\n'+selected[i]+'\n\"\"\"'}


for i in range(len(selected)):    
    completion = client.chat.completions.create(
      model= "gpt-3.5-turbo-1106",
      seed=0,
      temperature=0,
      messages= [
          {"role": "user", "content": 'What is the possibility that this is an interview? Give a short standard response in the first sentence. Explain in further sentences.\n Text: \"\"\"\n'+selected[i]+'\n\"\"\"'}
          ],
    )
    extralonganswer.append(completion.choices[0].message.content)


first="probability this is an interview is very high"
second="this is an interview."
third="probability is low"
fourth="probability is very low"

sent1=[x for x in extralonganswer if first in x.lower()]
sent2=[x for x in extralonganswer if second in x.lower()]
sent3=[x for x in extralonganswer if third in x.lower()]
sent4=[x for x in extralonganswer if third in x.lower()]

from nltk import sent_tokenize
first_sentences=[sent_tokenize(x)[0] for x in extralonganswer]

diction={"text": selected, "answer": extralonganswer, "first sentence": first_sentences}
classificationframe=pd.DataFrame(diction)

first=["possibility","is","high"]
sent1=[x for x in first_sentences if all(y in x.lower() for y in first)]
second=["possibility","is","low"]
sent2=[x for x in first_sentences if all(y in x.lower() for y in second)]
third="not an interview."
sent3=[x for x in first_sentences if third in x.lower()]
fourth=["not","possible","determine"]
sent4=[x for x in first_sentences if all(y in x.lower() for y in fourth)]
fifth=["unlikely"]
sent5=[x for x in first_sentences if all(y in x.lower() for y in fifth)]
sixth=["not","likely"]
sent6=[x for x in first_sentences if all(y in x.lower() for y in sixth)]
seventh=[" likely"]
sent7=[x for x in first_sentences if (all(y in x.lower() for y in seventh) and (not("not" in x.lower())))]
eith=["not","interview"]
sent8=[x for x in first_sentences if all(y in x.lower() for y in eith)]
ninth=" no possibility"

selectionframe=classificationframe.loc[classificationframe["first sentence"].apply(lambda x: not(all(y in x.lower() for y in eith)))]
selectionframe=selectionframe.loc[selectionframe["first sentence"].apply(lambda x: not(all(y in x.lower() for y in second)))]
selectionframe=selectionframe.loc[selectionframe["first sentence"].apply(lambda x: not(ninth in x.lower()))]
selectionframe=selectionframe.loc[selectionframe["first sentence"].apply(lambda x: "interview" in x.lower())]
selectionframe=selectionframe.loc[selectionframe["first sentence"].apply(lambda x: not(all(y in x.lower() for y in fifth)))]

selectionframe=selectionframe.reset_index(drop=True)

    
dictionary={"text": selected, "OPEN AI response": extralonganswer}
interviewframe=pd.DataFrame(dictionary)

os.makedirs("C:/Users/Alex/Desktop/interview inq", exist_ok=True)

writer = pd.ExcelWriter("C:/Users/Alex/Desktop/interview inq/OPENAI Classification 2.xlsx", engine='xlsxwriter')
writer.book.use_zip64()
interviewframe.to_excel(writer, merge_cells=False)
writer.close()


