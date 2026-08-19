from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch

from torch.utils import data
from torch.utils.data import DataLoader, TensorDataset

_DEFAULT_DATA_DIR = os.path.join(
    os.path.dirname(__file__),
    "..", "..", "..", "..",
    "data", "processed", "AEP", "dataset",
)


def dataload(filename):

    df = pd.read_csv(filename)
    return df

def train_val_test_split(data,test_ratio):
    #val_ratio = test_ratio / (1 - test_ratio)
    #X, Y = input_output_split(data, target_col)

    val_ratio = 0.10

    train_data, test_data = train_test_split(data, test_size=test_ratio, shuffle=False)
    train_data, val_data = train_test_split(train_data, test_size=val_ratio, shuffle=False)

    return train_data, val_data, test_data


def data_transform(data,option ='std'):
#data is numpy array
#option is set to std for standardization or minmax

    n = data.shape[0]

    if option == 'std' :

        #perform standardization of data
        miu = np.mean(data,axis = 0)
        sigma = np.std(data,axis=0,dtype=float)
        temp_data = data-np.tile(miu,(n,1))
        std_data = np.divide(temp_data,np.tile(sigma,(n,1)))

        return std_data, miu, sigma

    elif option == 'minmax':

        #perform min-max normalization
        max_val = np.max(data,0)
        #print(max_val)
        min_val = np.min(data,0)
        #print(min_val)
        rng = max_val-min_val
        norm_data = np.divide(data - np.tile(min_val,(n,1)),np.tile(rng,(n,1)))

        return norm_data, min_val, rng


def inv_trans(data,option,param1, param2):
    # apply inverse of standardization or normalization for 1-D column/row vector
    # option: standard or minmax normalization
    # params:list of parameters applied while normalization
    # data is 1-D column vector

    if option == "std":
         #perform standardization of data
        miu = param1
        sigma = param2
        inv_data = data*sigma + miu
        return inv_data
    else : #MinMax normalization
        #perform min-max normalization
        min_val = param1
        rng = param2
        inv_data = data*rng+min_val
        return inv_data


def test_data_transform(data,param1, param2, option ='minmax'):

    n = data.shape[0]
    if option == 'std' :
        miu = param1
        sigma = param2
        temp_data = data-np.tile(miu,(n,1))
        std_data = np.divide(temp_data,np.tile(sigma,(n,1)))

        return std_data, miu, sigma
    elif option == 'minmax':
        max_val = param1
        min_val = param2

        rng = max_val-min_val
        norm_data = np.divide(data - np.tile(min_val,(n,1)),np.tile(rng,(n,1)))

        return norm_data

# Prepare the data for RNN model such that the data is presented as (num_samples,seq_length,num_features)

def data_preparation(data,n_past,n_future):
    '''
    input:
        data :[data, time , context inputs, primary inputs, output]
        n_past : number of past steps to be used for prediction
        n_future :  number of steps ahead

    returns:
        context input (Z): 'hours_sin', 'hours_cos' ' weekday_sin' 'weekday_cos'
        primary input (X): 'lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4', 'T5', 'RH_5',
                  'T6', 'RH_6', 'T7', 'RH_7', 'T8', 'RH_8', 'T9', 'RH_9',T_out', 'RH_out', 'Visibility'
        ouput/target (Y): 'Appliances'

    '''
    n,m = data.shape
    print(m,n)

    k = n_future
    t = n_past

    input_data = []
    output_data = []
    context_data = []
    info_data = pd.DataFrame()

    for i in range(t, (n-k+1)):

        context_data.append(data.iloc[i-t:i,2:4]) # context features
        input_data.append(data.iloc[i-t:i, 4:m]) #next remaining attributes are sensor data
        output_data.append([data.iloc[i+k-1:i+k,m-1]])  #last column is the Appliance Energy
        info_data = pd.concat([info_data,data.iloc[i+k-1:i+k,0:2]], axis = 0) #first two columns are date and time

    X = np.array(input_data)
    Y = np.array(output_data)
    Z = np.array(context_data)
    U = info_data  #additional info (date, time) this is required for plots

    return X, Y, Z, U


class AEPTorchDataset:
    def __init__(self, data_dir: Optional[str] = None):
        self.data_dir = str(Path(data_dir or _DEFAULT_DATA_DIR).resolve())

    def dataload(self, filename):
        df = pd.read_csv(os.path.join(self.data_dir, filename))
        return df


    def load(self, filename):
        data =  self.dataload(filename)

        data.drop(['rv1', 'rv2'], axis = 1, inplace = True)
        data['date'] = data['date'].astype('datetime64[ns]')

        hr = data.date.dt.hour
        mn = data.date.dt.minute
        dt = data.date.dt.date
        tm = data.date.dt.time
        weekday = data.date.dt.weekday

        #day_mins = hr*60 + mn
        nsm = hr*3600 + mn*60                #number of seconds from midnight

        data['date'] = dt
        data['time'] = tm
        #data['mins'] = day_mins
        data['nsm'] = nsm
        data['weekday'] = weekday

        data = data[['date', 'time', 'weekday','nsm','lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3',
                    'RH_3', 'T4', 'RH_4', 'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8',
                    'RH_8', 'T9', 'RH_9', 'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed',
                    'Visibility', 'Tdewpoint','Appliances']]
        
        data['weekday_sin'] = np.sin(data['weekday'] * (2 * np.pi / 7))
        data['weekday_cos'] = np.cos(data['weekday'] * (2 * np.pi / 7))

        data.drop(columns =['weekday'], inplace = True)

        data = data[['date', 'time','weekday_sin', 'weekday_cos',
            'nsm','lights', 'T1', 'RH_1', 'T2', 'RH_2', 'T3', 'RH_3', 'T4', 'RH_4',
             'T5', 'RH_5', 'T6', 'RH_6', 'T7', 'RH_7', 'T8','RH_8', 'T9', 'RH_9',
             'T_out', 'Press_mm_hg', 'RH_out', 'Windspeed', 'Visibility', 'Tdewpoint','Appliances']]

        feature_list = list(data.columns)
        feature_list = feature_list[2:]

        train_data, val_data, test_data = train_val_test_split(data, 0.20)
        train_data.reset_index(inplace=True)
        val_data.reset_index(inplace=True)
        test_data.reset_index(inplace=True)

        norm_train_data, param1, param2 = data_transform(np.array(train_data[feature_list]), option = 'minmax')
        train_df_norm = pd.DataFrame(norm_train_data, columns = feature_list)

        norm_val_data, param1, param2 = data_transform(np.array(val_data[feature_list]), option = 'minmax')
        val_df_norm = pd.DataFrame(norm_val_data, columns = feature_list)

        norm_test_data, param1, param2 = data_transform(np.array(test_data[feature_list]),option = 'minmax')
        test_df_norm = pd.DataFrame(norm_test_data, columns = feature_list)


        df_train = pd.concat([ train_data[['date','time']], train_df_norm], axis = 1)
        #print(df_train.shape)

        df_val = pd.concat([val_data[['date','time']], val_df_norm], axis = 1)
        #print(df_val.shape)

        df_test = pd.concat([ test_data[['date','time']], test_df_norm], axis = 1)
        #print(df_test.shape)

        cols = ['T9', 'RH_6', 'RH_5', 'RH_3', 'RH_2', 'RH_1']
        df_train.drop(cols, axis = 1, inplace = True)
        df_val.drop(cols, axis = 1, inplace = True)
        df_test.drop(cols, axis = 1, inplace = True)

        X_train, Y_train, Z_train, U_train = data_preparation(df_train, 12, 1)
        X_val, Y_val, Z_val, U_val = data_preparation(df_val, 12, 1)
        X_test, Y_test, Z_test, U_test = data_preparation(df_test, 12, 1)

        norm_train_data_recompute, train_min, train_range = data_transform(
            np.array(train_data[feature_list]), option='minmax'
        )
        target_idx = feature_list.index('Appliances')
        self.target_min = train_min[target_idx]
        self.target_max = train_min[target_idx] + train_range[target_idx]
        self.train_min = train_min
        self.train_range = train_range
        self.feature_list = feature_list

        self.U_train = U_train
        self.U_val = U_val
        self.U_test = U_test

        self.Z_train = Z_train
        self.Z_val = Z_val
        self.Z_test = Z_test

        batch_size =256

        #transform the arrays into torch tensors
        train_features = torch.Tensor(X_train)
        train_targets = torch.Tensor(Y_train)
        train_cx_features = torch.Tensor(Z_train)

        val_features = torch.Tensor(X_val)
        val_targets = torch.Tensor(Y_val)
        val_cx_features = torch.Tensor(Z_val)

        test_features = torch.Tensor(X_test)
        test_targets = torch.Tensor(Y_test)
        test_cx_features = torch.Tensor(Z_test)

        # Store normalization params and context info for external access
        # param1 = min_val, param2 = range (max - min) from the TRAIN split
        # We need train-split params for proper inverse transform
        

        # train = TensorDataset(train_features,train_cx_features, train_targets)
        # val = TensorDataset(val_features, val_cx_features,val_targets)
        # test = TensorDataset(test_features, test_cx_features, test_targets)


        # train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
        # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
        # test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

        train = TensorDataset(train_features, train_targets)
        val = TensorDataset(val_features, val_targets)
        test = TensorDataset(test_features, test_targets)


        train_loader = DataLoader(train, batch_size=batch_size, shuffle=False, drop_last=True)
        val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, drop_last=True)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, drop_last=True)

        return train_loader, val_loader, test_loader