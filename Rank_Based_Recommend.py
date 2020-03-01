# -*- coding: utf-8 -*-
"""
Created on Sat Feb 29 23:44:39 2020

@author: chijioke Idoko
"""

# Import libraries and read in datasets from IBM watson studio of user_article
#interactions
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import project_tests as t
import pickle

#%matplotlib inline

df = pd.read_csv('data/user-item-interactions.csv')
df_content = pd.read_csv('data/articles_community.csv')
del df['Unnamed: 0']
del df_content['Unnamed: 0']

# Show df to get an idea of the data
df.head()


# Run this cell to map the user email to a user_id column and remove the email column

def email_mapper():
    coded_dict = dict()
    cter = 1
    email_encoded = []
    
    for val in df['email']:
        if val not in coded_dict:
            coded_dict[val] = cter
            cter+=1
        
        email_encoded.append(coded_dict[val])
    return email_encoded

email_encoded = email_mapper()
del df['email']
df['user_id'] = email_encoded

# show header
df.head()

#Rank-Based Recommendation System
def get_top_articles(n, df):
    '''
    Produces the topmost popular n article titles from the dataframe df
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    #first group by article_ids and sort from most popular down
    my_d = df.groupby('article_id', axis = 0).count().sort_values(by = ['title'], ascending = False)
    
    #list of top article ids
    top_a = list(my_d['title'].index[:n].values)
    
    #return the article names (titles) using the ids
    top_articles = df[df['article_id'].isin(top_a)]['title'].values
    
    return top_articles # Return the top article titles from df (not df_content)

def get_top_article_ids(n, df):
    '''
    Produces the topmost popular n article ids from the dataframe df
    INPUT:
    n - (int) the number of top articles to return
    df - (pandas dataframe) df as defined at the top of the notebook 
    
    OUTPUT:
    top_articles - (list) A list of the top 'n' article titles 
    
    '''
    #group by article ids and sort descending
    my_d = df.groupby('article_id', axis = 0).count().sort_values(by = ['title'], ascending = False)
    
    #return the indices of these articles
    top_a = list(my_d['title'].index[:n].values)
    return top_a # Return the top article ids