#!/usr/bin/env python
# coding: utf-8

# In[2]:


def dot_product(d1,d2):
    if len(d1)>len(d2):
        return dot_product(d2,d1)
    else:
        return sum(d1.get(f,0)*v for f,v in d2.items())


# In[5]:


def increment(d1,d2,scale):
    '''
    Implements d1 += scale * d2 for sparse vectors.
    @param dict d1: the feature vector which is mutated.
    @param float scale
    @param dict d2: a feature vector.

    NOTE: This function does not return anything, but rather
    increments d1 in place. We do this because it is much faster to
    change elements of d1 in place than to build a new dictionary and
    return it.
    '''
        
    for f,v in d2.items():
        d1[f]=d1.get(f,0)+scale*v


# In[6]:


def dict_multiply(d1,number):
    for f,v in d1.items():
        d1[f]=v*numbers
        

