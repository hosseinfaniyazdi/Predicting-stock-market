

import math
import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import ta
from finta import TA as Tanalysis
from sklearn.preprocessing import MinMaxScaler, StandardScaler

###########################################################################
#########################    DATA  PREPARATION    #########################
###########################################################################

# Read data from csv file
def readStockData():
  Stock_data_pd = pd.read_csv('CSVForDate.csv', index_col="Date")
  Stock_data_pd = Stock_data_pd.drop(Stock_data_pd.columns[[4]], axis=1)
  Stock_data_pd = Stock_data_pd.rename(str.lower, axis='columns')
  return Stock_data_pd

def saveStockData_toCSV(dataframe, name):
  dataframe.to_csv(f'{name}.csv')

Stock_data_pd = readStockData()

###########################################################################
#########################    DATA  PREPARATION    #########################
###########################################################################

dataType    = 'Only_Close'

Scaler      = MinMaxScaler(feature_range=(0,1))
# Scaler   = StandardScaler()

# All Indicator For Input Data Set
all_data_np     = Stock_data_pd.filter(['close']).values
all_data_npS    = Scaler.fit_transform(all_data_np)

# Day's to input
Past_Days = 60

X_Data = []
Y_Data = []
for i in range(Past_Days, len(all_data_npS)):
    X_Data.append(all_data_npS[i - Past_Days: i, 0])
    Y_Data.append(all_data_npS[i][0])

# Make Sync data -> Today's indicator data for tomorrow closing price
days            = 1     # Days Ahead

# Find Number of data set
Train_data_len  = math.ceil(len(X_Data) * 0.7) - 1
Test_data_len   = math.ceil(len(X_Data) * 0.2)

# Divide data to 3 part 
X_Data,     Y_Data      = np.array(X_Data),                                                 np.array(Y_Data)
X_Train,    Y_Train     = np.array(X_Data[:Train_data_len]),                                np.array(Y_Data[:Train_data_len])
X_Test,     Y_Test      = np.array(X_Data[Train_data_len:Train_data_len + Test_data_len]),  np.array(Y_Data[Train_data_len:Train_data_len + Test_data_len])
X_Predict,  Y_Predict   = np.array(X_Data[Train_data_len + Test_data_len:]),                np.array(Y_Data[Train_data_len + Test_data_len:])

###########################################################################
#########################    ADDING  LIBRARIES    #########################
###########################################################################
# Libraries
from sklearn.svm import SVR
from sklearn.metrics import mean_absolute_error, mean_squared_error

###########################################################################
#########################   CREATING SVR MODEL   ##########################
###########################################################################

gamma = 4
svr  = SVR(kernel = 'rbf', gamma = gamma)

###########################################################################
############################  TRAINING PHASE  #############################
###########################################################################

svr.fit(X_Train, Y_Train)

###########################################################################
############################   TESTING PHASE   ############################
###########################################################################

Predict_Y_Test      = svr.predict(X_Test)
Predict_Y_Predict   = svr.predict(X_Predict)

# RMSE and MSE and MAE
print(f"------------------------------")
print(f"Run with Gamma {gamma} and closing price")
print(f"---------------TESTING---------------")
test_Error_mse = mean_squared_error(Y_Test, Predict_Y_Test)
test_Error_mae = mean_absolute_error(Y_Test, Predict_Y_Test)
print(f"Prediction MSE value: {test_Error_mse} and RMSE value: {np.sqrt(test_Error_mse)} and MAE value: {test_Error_mae}")
print(f"Prediction MSE value: {test_Error_mse:.4e} and RMSE value: {np.sqrt(test_Error_mse):.4e} and MAE value: {test_Error_mae:.4e}")
print(f"---------------PREDICTION---------------")
predict_Error_mse = mean_squared_error(Y_Predict, Predict_Y_Predict)
predict_Error_mae = mean_absolute_error(Y_Predict, Predict_Y_Predict)
print(f"Prediction MSE value: {predict_Error_mse} and RMSE value: {np.sqrt(predict_Error_mse)} and MAE value: {predict_Error_mae}")
print(f"Prediction MSE value: {predict_Error_mse:.4e} and RMSE value: {np.sqrt(predict_Error_mse):.4e} and MAE value: {predict_Error_mae:.4e}")
print(f"------------------------------")