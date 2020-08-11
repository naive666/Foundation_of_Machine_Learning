#!/usr/bin/env python
# coding: utf-8

# In[54]:


import os
import random


# In[47]:


def read_data(text):
    with open(text,'r') as file:
        lines=file.read()
        #要删除一些特殊的符号：'${}()[].,:;+-*/&|<>=~" '
        symbols='${}()[].,:;+-*/&|<>=~"'
        table=lines.maketrans('','',symbols)
        lines=lines.translate(table).strip()
        lines=lines.replace('\n','')
        lines=lines.split(' ')
        while '' in lines:
            lines.remove('')
    return lines


# In[48]:


father_path='E:\Bloomberg ML\hw\hw3\hw3-svm\data'
def get_file(path,label):
    review=[]
    text_list=os.listdir(os.path.join(father_path,path))
    for text in text_list:
        r=read_data(os.path.join(os.path.join(father_path,path),text))
        data=(r,label)
        review.append(data)
    return review


# In[67]:


def shuffle(list1,list2):
    total_file=list1+list2
    random.shuffle(total_file)
    return total_file

