import numpy as np
import pandas as pd
import random
from random import sample
from sklearn.model_selection import train_test_split
import torch
import csv

# --------------------------------------------------------------------------#
test_size = 0.3
# --------------------------------------------------------------------------#

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

class SpiltData_ex1(object):
    def spilt(self, splitSeed):
        setup_seed(splitSeed)  # raw seed: 8
        list_save_trainandtestx = ['408-h1.csv','408-h2.csv','408-h3.csv']
        rate = [1, 2, 3]
        total = 0
        for i in range(len(rate)):
            total += rate[i]
        spilt_rate = []
        for i in range(len(rate)):
            spilt_rate.append(rate[i] / total)

        df = pd.read_excel('408-D4.xlsx')
        header = df.columns
        data = df.loc[:].values
        for num_of_file in range(0, len(list_save_trainandtestx)):
            data_w = open(list_save_trainandtestx[num_of_file], 'w', newline='')
            sample_1 = sample((range(0, len(df))), int(len(df) * spilt_rate[num_of_file]))
            csv_write = csv.writer(data_w, dialect='excel')
            csv_write.writerow(header)

            for i in sample_1:
                csv_write.writerow(data[i])
            data_w.close()
        return spilt_rate


class SpiltData_ex2(object):
    def spilt(self,sn):
        setup_seed(8)
        list_save_trainandtestx = ['408-h1.csv','408-h2.csv','408-h3.csv']
        spilt_num = sn

        df = pd.read_excel('408-D4.xlsx')
        header = df.columns
        data = df.loc[:].values
        for num_of_file in range(0, len(list_save_trainandtestx)):
            data_w = open(list_save_trainandtestx[num_of_file], 'w', newline='')
            sample_1 = sample((range(0, len(df))), int(spilt_num))
            csv_write = csv.writer(data_w, dialect='excel')
            csv_write.writerow(header)

            for i in sample_1:
                csv_write.writerow(data[i])
            data_w.close()



class GetDataSet(object):
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.copdDataSetConstruct()

    def copdDataSetConstruct(self):
        data = pd.read_csv('408-h1.csv',encoding='gbk')

        x = data.drop(['id','level'], axis=1)

        y = data['level']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=5)  # random_state 随机种子
        x_tr = x_train.loc[:].values
        y_tr = y_train.loc[:].values
        x_te = x_test.loc[:].values
        y_te = y_test.loc[:].values

        self.train_data = x_tr
        self.train_label = y_tr
        self.test_data = x_te
        self.test_label = y_te

class GetDataSet2(object):
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.copdDataSetConstruct()

    def copdDataSetConstruct(self):
        data = pd.read_csv('408-h2.csv',encoding='gbk')

        x = data.drop(['id','level'], axis=1)

        y = data['level']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=6)  # random_state 随机种子
        x_tr = x_train.loc[:].values
        y_tr = y_train.loc[:].values
        x_te = x_test.loc[:].values
        y_te = y_test.loc[:].values

        self.train_data = x_tr
        self.train_label = y_tr
        self.test_data = x_te
        self.test_label = y_te

class GetDataSet3(object):
    def __init__(self):
        self.train_data = None
        self.train_label = None
        self.test_data = None
        self.test_label = None
        self.copdDataSetConstruct()

    def copdDataSetConstruct(self):
        data = pd.read_csv('408-h3.csv',encoding='gbk')

        x = data.drop(['id','level'], axis=1)

        y = data['level']

        x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=test_size, random_state=11)  # random_state is random seed.
        x_tr = x_train.loc[:].values
        y_tr = y_train.loc[:].values
        x_te = x_test.loc[:].values
        y_te = y_test.loc[:].values

        self.train_data = x_tr
        self.train_label = y_tr
        self.test_data = x_te
        self.test_label = y_te





