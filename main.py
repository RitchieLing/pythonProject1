# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
import csv
import io

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestRegressor
from sklearn.datasets import make_regression
from Helper import Model
import os





def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press ⌘F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    df_train = pd.read_csv('train.csv', index_col= [0])
    df_predict = pd.read_csv('test.csv', index_col=[0])

    model = Model(df_train,df_predict)
    data = model.train_and_predict()
    dataset = pd.DataFrame({'price': data})
    dataset.index.name = 'id'
    print(dataset)
    dataset.to_csv('outptut.csv',index = True)




# See PyCharm help at https://www.jetbrains.com/help/pycharm/
