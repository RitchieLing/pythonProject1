import csv
import numpy as np
import csv
import io

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression




class Model():

    def __init__(self,df_train,df_predict):
        self.df_train = df_train
        self.df_predict = df_predict
        dict_map = {'E': int(6), 'H': int(3),
                    'D': int(7),'G': int(4),
                    'F' : int(5),'I':int(2),
                    'J':int(1)}
        updateSer = self.df_train['color'].map(dict_map)
        self.df_train['color'] = updateSer
        ## work on cut
        dict_map = {'Ideal': int(5), 'Premium': int(4),
                    'Very Good': int(3), 'Good': int(2),
                    'Fair': int(1)}
        updateSer = self.df_train['cut'].map(dict_map)
        self.df_train['cut'] = updateSer
        ##work on clarity
        dict_map = {'IF': int(8), 'VVS1': int(7),
                    'VVS2': int(6), 'VS1': int(5),
                    'VS2': int(4), 'SI1': int(3),
                    'SI2': int(2), 'I1': int(1),
                    }
        updateSer = self.df_train['clarity'].map(dict_map)
        self.df_train['clarity'] = updateSer

        ####################################
        ########## predict ################
        dict_map = {'E': int(6), 'H': int(3),
                    'D': int(7), 'G': int(4),
                    'F': int(5), 'I': int(2),
                    'J': int(1)}
        updateSer = self.df_predict['color'].map(dict_map)
        self.df_predict['color'] = updateSer

        dict_map = {'Ideal': int(5), 'Premium': int(4),
                    'Very Good': int(3), 'Good': int(2),
                    'Fair': int(1)}
        updateSer = self.df_predict['cut'].map(dict_map)
        self.df_predict['cut'] = updateSer
        ##work on clarity
        dict_map = {'IF': int(8), 'VVS1': int(7),
                    'VVS2': int(6), 'VS1': int(5),
                    'VS2': int(4), 'SI1': int(3),
                    'SI2': int(2), 'I1': int(1),
                    }
        updateSer = self.df_predict['clarity'].map(dict_map)
        self.df_predict['clarity'] = updateSer



    def train_and_predict(self):
        regr = RandomForestRegressor(max_depth=10, random_state=0)
        y = self.df_train['price']
        print(y)
        self.df_train.drop(['price'], inplace=True, axis=1)

        Y = y.to_numpy()
        print(Y)
        X = self.df_train.to_numpy()
        print(X)
        regr.fit(X, y)
        print(self.df_predict.head())
        X_hat = self.df_predict.to_numpy()
        return regr.predict(X_hat)

