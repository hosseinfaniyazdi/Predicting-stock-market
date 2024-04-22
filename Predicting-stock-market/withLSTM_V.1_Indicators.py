

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

###########################################################################
#########################     STOCK INDICATOR     #########################
###########################################################################

def SMA(dataframe, period, use=1, close='close'):
    """
    Simple moving average - rolling mean in pandas lingo. Also known as 'MA'.
    The simple moving average (SMA) is the most basic of the moving averages used for trading.
    :return your datafram
    """
    if use == 1:
        sma = dataframe[close].rolling(window=period).mean()

    elif use == 2:
        sma = ta.trend.SMAIndicator(close=dataframe[close], window=period)
        sma = sma.sma_indicator()

    elif use == 3:
        sma = Tanalysis.SMA(dataframe, period, column=close)

    dataframe[f'SMA{period}'] = sma
    indicatorName.append(f'SMA{period}')
    return dataframe


def EMA(dataframe, period, use=1, close='close'):
    """
    Exponential Weighted Moving Average - Like all moving average indicators, they are much better suited for trending markets.
    When the market is in a strong and sustained uptrend, the EMA indicator line will also show an uptrend and vice-versa for a down trend.
    EMAs are commonly used in conjunction with other indicators to confirm significant market moves and to gauge their validity.
    :return your datafram
    """
    if use == 1:
        ema = dataframe[close].ewm(span=period, adjust=False).mean()

    elif use == 2:
        ema = ta.trend.EMAIndicator(close=dataframe[close], window=period)
        ema = ema.ema_indicator()

    elif use == 3:
        ema = Tanalysis.EMA(dataframe, period, column=close)

    dataframe[f'EMA{period}'] = ema
    indicatorName.append(f'EMA{period}')
    return dataframe


def RSI(dataframe, period=14, use=1, close='close'):
    """
    Relative Strength Index (RSI) is a momentum oscillator that measures the speed and change of price movements.
    RSI oscillates between zero and 100. Traditionally, and according to Wilder, RSI is considered overbought when above 70 and oversold when below 30.
    Signals can also be generated by looking for divergences, failure swings and centerline crossovers.
    RSI can also be used to identify the general trend.
    :return your datafram
    """
    if use == 1:
        delta = dataframe[close].diff()

        dUp, dDown = delta.copy(), delta.copy()
        dUp[dUp < 0] = 0
        dDown[dDown > 0] = 0

        # RolUp = dUp.rolling(window = period).mean()
        RolUp = dUp.ewm(alpha=1.0 / period).mean()
        # RolDown = dDown.rolling(window = period).mean().abs()
        RolDown = dDown.abs().ewm(alpha=1.0 / period).mean()

        RS = RolUp / RolDown
        rsi = 100.0 - (100.0 / (1.0 + RS))

    elif use == 2:
        rsi = ta.momentum.rsi(close=dataframe[close], window=period)

    elif use == 3:
        rsi = Tanalysis.RSI(dataframe, period, column=close)

    dataframe[f"RSI{period}"] = rsi
    indicatorName.append(f"RSI{period}")
    return dataframe


def CCI(dataframe, period=20, constant=0.015, use=1, low='low', high='high', close='close'):
    """
    Commodity Channel Index (CCI) is a versatile indicator that can be used to identify a new trend or warn of extreme conditions.
    CCI measures the current price level relative to an average price level over a given period of time.
    The CCI typically oscillates above and below a zero line. Normal oscillations will occur within the range of +100 and −100.
    Readings above +100 imply an overbought condition, while readings below −100 imply an oversold condition.
    As with other overbought/oversold indicators, this means that there is a large probability that the price will correct to more representative levels.

    source: https://stockcharts.com/school/doku.php?id=chart_school:technical_indicators:commodity_channel_index_cci

    :param pd.DataFrame ohlc: 'open, high, low, close' pandas DataFrame
    :period: int - number of periods to take into consideration
    :factor float: the constant at .015 to ensure that approximately 70 to 80 percent of CCI values would fall between -100 and +100.
    :return your datafram
    """
    if use == 1:
        TP = (dataframe[high] + dataframe[low] + dataframe[close]) / 3
        CCI = (TP - TP.rolling(window=period).mean()) / \
            (constant * TP.rolling(window=period).std())

    elif use == 2:
        CCI = ta.trend.cci(high=dataframe[high], low=dataframe[low],
                           close=dataframe[close], window=period, constant=constant)

    elif use == 3:
        CCI = Tanalysis.CCI(dataframe, period, constant)

    dataframe[f'CCI{period}'] = CCI
    indicatorName.append(f'CCI{period}')
    return dataframe


def MACD(dataframe, period_fast=12,  period_slow=26,  period_sign=9, use=3, close='close'):
    if use == 1:
        pass

    elif use == 2:
        macd = ta.trend.MACD(
            close=dataframe[close], window_fast=period_fast, window_slow=period_slow, window_sign=period_sign)
        dataframe[f'MACD'] = macd.macd()
        dataframe[f'MACD_HISTOGRAM'] = macd.macd_diff()
        dataframe[f'MACD_SIGNAL'] = macd.macd_signal()

    elif use == 3:
        macd = Tanalysis.MACD(dataframe, period_fast,
                              period_slow, period_sign, column=close)
        dataframe[f'MACD'] = macd['MACD']
        dataframe[f'MACD_SIGNAL'] = macd['SIGNAL']

    indicatorName.append(f'MACD')
    indicatorName.append(f'MACD_SIGNAL')
    return dataframe


def WilliamsR(dataframe, period=14, use=3, low='low', high='high', close='close'):
    if use == 1:
        pass

    elif use == 2:
        wr = ta.momentum.WilliamsRIndicator(
            high=dataframe[high], low=dataframe[low], close=dataframe[close], lbp=period)
        wr = wr.williams_r()

    elif use == 3:
        wr = Tanalysis.WILLIAMS(dataframe, period=period)

    dataframe[f'Williams_%R{period}'] = wr
    indicatorName.append(f'Williams_%R{period}')
    return dataframe


def Momentum(dataframe, period=10, use=3, close='close'):
    if use == 1:
        pass

    elif use == 2:
        pass
    elif use == 3:
        mom = Tanalysis.MOM(dataframe, period, column=close)

    dataframe[f'MOM{period}'] = mom
    indicatorName.append(f'MOM{period}')
    return dataframe


def Stochastics(dataframe, k, d, use=3, low='low', high='high', close='close'):
    """
    Fast stochastic calculation
    %K = (Current Close - Lowest Low)/
    (Highest High - Lowest Low) * 100
    %D = 3-day SMA of %K

    Slow stochastic calculation
    %K = %D of fast stochastic
    %D = 3-day SMA of %K

    When %K crosses above %D, buy signal 
    When the %K crosses below %D, sell signal
    """
    if use == 1:
        df = dataframe.copy()

        # Set minimum low and maximum high of the k stoch
        low_min = df[low].rolling(window=k).min()
        high_max = df[high].rolling(window=k).max()

        # Fast Stochastic
        dataframe[f'St_%K{k}_Fast'] = 100 * \
            (df[close] - low_min)/(high_max - low_min)
        dataframe[f'St_%D{d}_Fast'] = dataframe[f'St_%K{k}_Fast'].rolling(
            window=d).mean()

        # Slow Stochastic
        dataframe[f'St_%K{k}_Slow'] = dataframe[f'St_%D{d}_Fast']
        dataframe[f'St_%D{d}_Slow'] = dataframe[f'St_%K{k}_Slow'].rolling(
            window=d).mean()

    elif use == 2:
        pass
    elif use == 3:
        dataframe[f'St_%K{k}_Fast'] = Tanalysis.STOCH(dataframe, period=k)
        dataframe[f'St_%D{d}_Fast'] = Tanalysis.STOCHD(
            dataframe, period=d, stoch_period=k)

    indicatorName.append(f'St_%K{k}_Fast')
    indicatorName.append(f'St_%D{d}_Fast')
    return dataframe


def AccDistIndicator(dataframe, use=3, low='low', high='high', close='close', volume='volume'):
    if use == 1:
        pass
    elif use == 2:
        aci = ta.volume.AccDistIndexIndicator(
            high=dataframe[high], low=dataframe[low], close=dataframe[close], valume=dataframe[volume])
        aci = aci.acc_dist_index()
    elif use == 3:
        aci = Tanalysis.ADL(dataframe)

    dataframe[f'ACI'] = aci
    indicatorName.append(f'ACI')
    return dataframe


###########################################################################
########################   RUN & ADD  INDICATORS   ########################
###########################################################################

indicatorName = []
Stock_data_pd = readStockData()
# (use = 1) -> selfwrite
# (use = 2) -> ta library
# (use = 3) -> finta library
SMA(Stock_data_pd, 10, use = 3)
EMA(Stock_data_pd, 50, use = 3)
RSI(Stock_data_pd, 50, use = 3)
CCI(Stock_data_pd, 20, 0.015, use = 3)
MACD(Stock_data_pd, period_fast = 12,  period_slow = 26,  period_sign = 9, use = 3)
WilliamsR(Stock_data_pd, period = 14, use = 3)
Momentum(Stock_data_pd, period = 10, use = 3)
Stochastics(Stock_data_pd, 14, 3, use = 3)
# AccDistIndicator(Stock_data_pd, use = 3)
saveStockData_toCSV(Stock_data_pd, 'With_Indicators')
print(indicatorName)


###########################################################################
#########################    DATA  PREPARATION    #########################
###########################################################################

dataType    = 'Indicators'

Scaler      = MinMaxScaler(feature_range=(0,1))
# Scaler   = StandardScaler()

list_name = []
list_name.extend(indicatorName)

# All Indicator For Input Data Set
all_data_np     = Stock_data_pd.filter(list_name).values
all_data_np     = all_data_np[50:] 
all_data_npS    = Scaler.fit_transform(all_data_np)

# Only Close 
close_data_np   = Stock_data_pd.filter(['close']).values
close_data_np   = close_data_np[50:] 
close_data_npS  = Scaler.fit_transform(close_data_np)

# Make Sync data -> Today's indicator data for tomorrow closing price
days            = 1     # Days Ahead
all_data_npS    = all_data_npS[:-days]
close_data_npS  = close_data_npS[days:]

# Find Number of data set
Train_data_len  = math.ceil(len(all_data_npS) * 0.7) - 1
Test_data_len   = math.ceil(len(all_data_npS) * 0.2)

# Divide data to 3 part 
X_Train,    Y_Train     = np.array(all_data_npS[:Train_data_len]),                                np.array(close_data_npS[:Train_data_len])
X_Test,     Y_Test      = np.array(all_data_npS[Train_data_len:Train_data_len + Test_data_len]),  np.array(close_data_npS[Train_data_len:Train_data_len + Test_data_len])
X_Predict,  Y_Predict   = np.array(all_data_npS[Train_data_len + Test_data_len:]),                np.array(close_data_npS[Train_data_len + Test_data_len:])

# Reshape X-data from nd to 3d 
X_Train     = np.reshape(X_Train, (X_Train.shape[0], X_Train.shape[1], 1))
X_Test      = np.reshape(X_Test, (X_Test.shape[0], X_Test.shape[1], 1))
X_Predict   = np.reshape(X_Predict, (X_Predict.shape[0], X_Predict.shape[1], 1))

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

layerName = 'LSTM_Dense'

# Designing the Neural
model = Sequential()
model.add(LSTM(50, return_sequences = True , input_shape = (X_Train.shape[1] , 1) , name='layer_1'))
model.add(LSTM(100, return_sequences = False , name='layer_2'))
model.add(Dense(50 , name='layer_3', activation='relu'))
model.add(Dense(1 ,  name='layer_4', activation='relu'))

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
