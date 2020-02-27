# -*- coding: utf-8 -*-
"""
Created on Thu Feb 27 12:25:19 2020

@author: chiji
"""

class SVD_():
    
    
    """ 
    """
    def __init__(self, df):
        self.df = df

    def create_test_and_train_user_item(df_train, df_test):
        '''
        creats test and train sets of a user_item matrix fron a given dataframe
        
        INPUT:
            df_train - training dataframe
            df_test - test dataframe
            
            OUTPUT:
                - user_item_train - a user-item matrix of the training dataframe 
                (unique users for each row and unique articles for each column)
                user_item_test - a user-item matrix of the testing dataframe 
                (unique users for each row and unique articles for each column)
                test_idx - all of the test user ids
                test_arts - all of the test article id
        '''
        #Using the function 'create_user_item_matrix' to create user_item train annd test
        user_item_train = create_user_item_matrix(df_train)
        user_item_test = create_user_item_matrix(df_test)
        test_idx = list(user_item_test.index)#userss in test set
        test_arts = list(user_item_test.columns)#articles in test set
    
        return user_item_train, user_item_test, test_idx, test_arts
    
    def find_predictables(user_item_train, user_item_test):
        '''
        Find the common users in the test and train user_item matrices to which predictions can be made
        
        Input: user_item_train - the train matrix of user_item interactions
                user_item_test - the test matrix of user_item interactions
        OUTPUT:
            common_ids - the ids of the coomon users among the train and test user_item matrices
            common_user_item_test - the arrray of user_item common to both test and train sets
        '''
        #The common users among the train and test user_items
        common_user_item_test=user_item_train[user_item_train.index.isin(user_item_test.index)]
        common_ids = common_user_item_test.index#common ids in the train and test sets
        return common_ids, common_user_item_test
    def fit(user_item_train):
        '''
        Input: User_item_train - the train set to decompose into three matrices
        Output - the results of decomposing a user_item matrice
            - u_train
            - s_train
            - vt_train
        
        '''
        # fit SVD on the user_item_train matrix
        u_train, s_train, vt_train = np.linalg.svd(user_item_train)
        return u_train, s_train, vt_train
    
    def predict(user_item_train, user_item_test, u_matrix, vt_matrix, s_matrix, k):
        '''
        Make possible predictions on the test dataset, given a train dataset of the user_item interactions using SVD
        INPUT:
        user_item_train - train set of user_item interactions
        user_item_test - test set of user_item interactions
        user_matrix - user by latent factor matrix
        vt_matrix - latent factor by item matrix
        s_matrix - latent factors
        k - the number of latent factors to use in making predictions
        
        OUTPUT:
        user_item_est - the predictions on user_item_test according to SVD
        '''
        
        #Map out the appropriate indices of the common users from the user_item_train matrix and use
        #it to obtain corrresponding arrays in the u_matrix (u_train). Return a concatenation of these arrys
        #that will help in predicting in the test set
        
        user_ids_series = np.array(user_item_train.index)#series of user_ids from the user_item train
        user_row = np.where(user_ids_series == common_ids[0])[0][0]#Pick the row of that user
        u_train_i = np.array([u_matrix[user_row]])# get the corresponding array from u_train
        
        #find the users's indices that exist in both train and and test dataframe
        for i in common_ids[1:]:
            user_row = np.where(user_ids_series == i)[0][0]#Pick the row of that user
            arr_i = np.array([u_matrix[user_row]])#get the corresponding array from u_train
            u_train_i = np.concatenate((u_train_i, arr_i), axis = 0)
        
        # restructure with k latent features
        s_new, u_new, vt_new = np.diag(s_train[:k]), u_train_i[:, :k], vt_train[:k, :]
        
        # take dot product
        user_item_est = np.around(np.dot(np.dot(u_new, s_new), vt_new)))
        #pred = np.dot(np.dot(u_2, s_2), vt_2)
        return user_item_est