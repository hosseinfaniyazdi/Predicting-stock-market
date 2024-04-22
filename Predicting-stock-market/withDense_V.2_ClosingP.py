

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
import os
from datetime import datetime
from pathlib import Path

import keras
import matplotlib.pyplot as plt
from keras.layers import *
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

###########################################################################
#########################   CREATING NEURAL NET   #########################
###########################################################################

layerName = 'Dense'

# Designing the Neural
model = Sequential()
model.add(Dense(50, input_dim = Past_Days, activation = 'sigmoid', name='layer_1'))
model.add(Dense(1 ,  activation='relu', name='layer_2'))

# Compiling and Get Summary
model.compile(optimizer = 'adam' , loss = 'mse')
model.summary()

###########################################################################
##########################    TRAINING  PHASE    ##########################
###########################################################################

# Log Details
def logging_details(epochs, shuffle, validation_split, days, current_time, current_date, layerName):
  print(f"Run on {current_date}, {current_time} - with {epochs} Epochs, {layerName} Layer, Shuffle {shuffle} and Validation split on train data {validation_split} ")

# Trian Setting
epochs            = 50
verbose           = 1
shuffle           = True
validation_split  = 0.2

# Data and Time
now                   = datetime.now()
current_time          = now.strftime("%H.%M.%S")
current_date          = now.strftime("%d.%m.%Y")

# Log Details
logging_details(epochs, shuffle, validation_split, days, current_time, current_date, layerName)

# We Can Save The Log
saveDataDirRoot       = "SaveData"
saveDataDir           = f"SaveData/{current_date}"
saveLogDir            = "TrainLogs"

if not os.path.exists(f"{saveDataDir}/{saveLogDir}"):
  os.mkdir(saveDataDirRoot)
  os.mkdir(saveDataDir)
  os.mkdir(f"{saveDataDir}/{saveLogDir}")

# Create a TensorBoard logger (If you want)
# On Your System use \ for address in this part
# On Google Colab use /
saveDataDirlog = f"SaveData\{current_date}"
direction   = f"{saveDataDirlog}\{saveLogDir}"
RUN_NAME    = f"{current_time} - Build on {epochs} epochs,{days} Days ahead, {layerName} Layer, {dataType} Shf-{shuffle}, V_sp-{validation_split}"
logger      = keras.callbacks.TensorBoard(
    log_dir           = '{}\{}'.format(direction, RUN_NAME),
    write_graph       = True,
    histogram_freq    = 5,
)

# Training action
model.fit(  x                 = X_Train, 
            y                 = Y_Train, 
            batch_size        = 1,
            verbose           = verbose,
            epochs            = epochs,
            shuffle           = shuffle,
            validation_split  = validation_split,
            callbacks         = [logger]
        )

# We Can Save the Lesult and Model
saveModelDir    = "ModelSave"

if not os.path.exists(f"{saveDataDir}/{saveModelDir}"):
    os.mkdir(f"{saveDataDir}/{saveModelDir}")

# Save Model Sturcture
model_structure = model.to_json()
path            = f"Build on {epochs} epochs,{days} Days, {current_time} - Train"
if not os.path.exists(f"{saveDataDir}/{saveModelDir}/{path}"):
    os.mkdir(f"{saveDataDir}/{saveModelDir}/{path}")
f = Path(f"{saveDataDir}/{saveModelDir}/{path}/Model_Structure.json")
f.write_text(model_structure)

# Save the Trained Model
if not os.path.exists(f"{saveDataDir}/{saveModelDir}/{path}"):
    os.mkdir(f"{saveDataDir}/{saveModelDir}/{path}")
model.save(f"{saveDataDir}/{saveModelDir}/{path}/Model.h5")

###########################################################################
############################   LOADING PHASE   ############################
###########################################################################

# Load Saved Model
saveDataDir = f"SaveData/{current_date}"
path        = "Build on 1000 epochs,1 Days, 15.43.12 - Train"
modelDir    = f"{saveDataDir}/{saveModelDir}/{path}/Model.h5"
# model = load_model(modelDir)

###########################################################################
############################   TESTING PHASE   ############################
###########################################################################

print(f"---------------TEST---------------")
test_Error_MSE = model.evaluate(x        = X_Test,
                                y        = Y_Test,
                                verbose  = 2)

# RMSE and MSE and Accuracy
test_Error_RMSE  = np.sqrt(test_Error_MSE) 
print(f"The MSE for Test data set is: {test_Error_MSE} , RMSE Value: {test_Error_RMSE}")
print(f"The MSE for Test data set is: {test_Error_MSE:.4e} , RMSE Value: {test_Error_RMSE:.4e}")
print(f"------------------------------")

###########################################################################
############################   PREDICT PHASE   ############################
###########################################################################

# Predict the 20% last of data for Asure
Pure_Predictions = model.predict(X_Predict)

# re-Sacle Predict data
# Predictions = Scaler.inverse_transform(Pure_Predictions)
Predictions = Pure_Predictions

# RMSE and MSE and MAE
print(f"---------------PREDICTION---------------")
predict_Error_mse   = np.mean((Predictions - Y_Predict) ** 2) 
predict_Error_rmse  = np.sqrt(np.mean((Predictions - Y_Predict) ** 2))
print(f"Prediction MSE value: {predict_Error_mse} and RMSE value: {predict_Error_rmse}")
print(f"Prediction MSE value: {predict_Error_mse:.4e} and RMSE value: {predict_Error_rmse:.4e}")
print(f"---------------")
predict_Error_mse = mean_squared_error(Y_Predict, Predictions)
predict_Error_mae = mean_absolute_error(Y_Predict, Predictions)
print(f"Prediction MSE value: {predict_Error_mse} and RMSE value: {np.sqrt(predict_Error_mse)} and MAE value: {predict_Error_mae}")
print(f"Prediction MSE value: {predict_Error_mse:.4e} and RMSE value: {np.sqrt(predict_Error_mse):.4e} and MAE value: {predict_Error_mae:.4e}")
print(f"------------------------------")
