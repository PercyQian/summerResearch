# %%
import pandas as pd 
import matplotlib.pyplot as plt 
%matplotlib inline 
import numpy as np
import random
import time 
import datetime as dt 
import os

# Summer2024-env/Scripts/activate

# %%
import seaborn as sns


# %%
# Url of weather data 

# %%
t = pd.read_csv("newWeatherData.csv")

# %%
t = t[5:len(t)-20]

# %%
t[:739].iloc[5]['time'].split("T")[1].split(":")[0]

# %%
t

# %%
def addWeatherHourColumn(data : pd.DataFrame) -> pd.DataFrame:

    data = data.reset_index(drop=True)

    data["hourTime"] = "0"

    # use datetime module 
    for index, row in data.iterrows():
        AMPM = "AM"
        hour = int(row['time'].split("T")[1].split(":")[0])

        if hour >12:
            AMPM = "PM"
            hour -=12
        
        # Note that there is no hour 24. After hour 23, it goes back to hour 0
        if hour == 12:
            AMPM = "PM"
        
        # 0 represents 12AM
        if hour == 0:
            hour = 12
            AMPM = "AM"


        data.loc[index,'hourTime'] = str(hour) + str(AMPM)

    return data

# %%
t

# %%
# Push to repo 

# %%
t= addWeatherHourColumn(t)

# %%
t

# %%
def saveWeatherHourlyData(data : pd.DataFrame , path : str) -> pd.DataFrame:

    hours = list(data['hourTime'].unique())

    for hour in hours:
        PATH = "hourlyWeatherData/" + path + "/" "weather_data_" + str(hour) + ".csv"

        

        try:
            pd.read_pickle(PATH)
        except Exception as e:    
            data[data['hourTime'] == hour].reset_index(drop=True).to_pickle(PATH)

# %%
#saveWeatherHourlyData(t,"openMeteo")

# %%


# %%


# %%
t.head()

# %%
months = ["janurary","february", "march", "april", "may", "june", "july", "august", "september", "october", "november","december"]

years = ["2021","2022","2023","2024"]
'''weatherData = pd.DataFrame(columns=["Date","Max Temp","Min Temp","Mean Temp","Departure","HDD","CDD","Precipitation","New Snow",'Snow Depth'])
    
for year in years:
    for month in months:
        print(year ,month)
        
        try:
            weatherData = pd.concat([weatherData , pd.read_csv("weatherData/" +month + "_" + year + "_weather_data.csv")] )
        except Exception as e:
            print("Ended on " + month + " " + year)

        print("Len of dataset",len(weatherData))
        print()

            
'''

# %%


# %%
# BIG TIME : 6 AM to 10 PM. Action is in the middle of the day 
# Hourly Weather DataSet (If it can be found ) or copy exact value for each of the 24 hours 
# Use hour as a feature for predictions so model can switch bettwen 
# Day of the week 
# Holidays (feature)

# %%
'''westernData = pd.DataFrame(columns=list(pd.read_csv("lmp_data/janurary_2021_lmp_data.csv").columns)) 
westIntData = pd.DataFrame(columns=list(pd.read_csv("lmp_data/janurary_2021_lmp_data.csv").columns))

for year in years:
    for month in months:
        print(year ,month)
        
        try:
            hold = pd.read_csv("lmp_data/" +month + "_" + year + "_lmp_data.csv")
            westIntData = pd.concat([westIntData , ( hold[hold['pnode_name'] == "WEST INT HUB"][::-1] )] )
            westernData = pd.concat([westernData , ( hold[hold['pnode_name'] == "WESTERN HUB"][::-1] )])
        except Exception as e:
            print(e)
            print("Ended on " + month + " " + year)
            break

        print("Len of westIntData",len(westIntData))
        print("Len of westernData",len(westernData))

        print()'''

# %%
#westIntData

# %%
#westernData.info()

# %%
'''## Dropping null columns 
westernData.drop(["voltage", "equipment", "zone"], axis=1, inplace=True)
westIntData.drop(["voltage", "equipment", "zone"], axis=1, inplace=True)

# There appears to be a 100% correlation between the pnode_id and pnode_name so I will drop the pnode_name. 
westIntData.drop(["pnode_name", "type", "pnode_id"], axis=1, inplace=True)
westernData.drop(["pnode_name", "type", "pnode_id"], axis=1, inplace=True)'''

# %%
#westernData = westernData.reset_index(drop=True)

# %%
def addDaysColumn(data : pd.DataFrame) -> pd.DataFrame:

    data['currentDay'] = "0"
    data = data.reset_index(drop=True)
    
    # Janurary 1 2021 is a friday hence the mapping starts as friday equaling 1 and since indexes start at 0, thursday is 0
    savedDay = 1
    days = {
        1 : "friday",
        2: "saturday",
        3: "sunday",
        4: "monday",
        5: "tuesday",
        6: "wednesday",
        0: "thursday"
    }

    for index, row in data.iterrows():

        try:
            currentDay = int(row['datetime_beginning_ept'].split(" ")[0].split("/")[1]) % 7  # mod 7 to keep the days repeating
            data.loc[index,'currentDay'] = str(days[currentDay])
        except Exception as e:
            print(e)
            print(index)
            print(row)

    return data


# %%
def addHourColumn(data : pd.DataFrame) -> pd.DataFrame:

    data = data.reset_index(drop=True)

    data["hourTime"] = "0"

    # use datetime module 
    for index, row in data.iterrows():
        hour = row["datetime_beginning_utc"].split(" ")[1].split(":")[0]
        AMPM = row["datetime_beginning_utc"].split(" ")[2]

        data.loc[index,'hourTime'] = str(hour) + str(AMPM)

    return data

# %%
def saveHourlyData(data : pd.DataFrame , path : str) -> pd.DataFrame:

    hours = list(data['hourTime'].unique())

    for hour in hours:
        PATH = "hourlyLmpData/" + path + "/" "lmp_data_" + str(hour) + ".csv"

        

        try:
            pd.read_pickle(PATH)
        except Exception as e:    
            addDaysColumn(data)[data['hourTime'] == hour].reset_index(drop=True).to_pickle(PATH)

# %%
#saveHourlyData(addHourColumn(westernData), path="westernData")
#saveHourlyData(addHourColumn(westIntData), path="westIntData")

# %%
#addDaysColumn(westernData)

# %%
# Add holidays feature


# %%
'''
New Year's Day: January 1
Juneteenth: June 19
Independence Day: July 4
Veterans Day: November 11
Christmas Day: December 25

Martin Luther King Jr. Day: Third Monday in January
Presidents' Day: Third Monday in February
Memorial Day: Last Monday in May
Labor Day: First Monday in September
Columbus Day: Second Monday in October
Thanksgiving Day: Fourth Thursday of November
'''



# %%
def applyHoliday(data : pd.DataFrame, holiday: list[dict]) -> None:

    mapping = {
        1:"monday",
        2:"tueday",
        3:"wednesday",
        4:"thursday",
        5:"friday",
        6:"saturday",
        7:"sunday",

    }

    data = data.reset_index(drop=True)

    data['isHoliday'] = 0 # 0 for not holiday, 1 for is holiday

    for item in holiday:
        month = item['month']
        day = item['date']
        occurence = item['occurence']
        startYear = item['startYear']
        
        
        # occurence refers to when the day occurs. Ex is it the first monday occurence of the month or the second monday occurence of the month 
        occurence_match = 1 




        for index, row in data.iterrows():
            current_month = int(row["datetime_beginning_utc"].split(" ")[0].split("/")[0])
            current_day = int(row["datetime_beginning_utc"].split(" ")[0].split("/")[1])
            current_year = int(row["datetime_beginning_utc"].split(" ")[0].split("/")[2])


            if current_month == month and startYear <= current_year:


                try:
                    day = int(day)
                    if current_day == day:
                        data.loc[index, "isHoliday"] = 1
                                    
                except Exception as e:
                    
                    if mapping[dt.datetime(current_year, current_month, current_day).isocalendar().weekday] == str(day):
                        if occurence_match == occurence:
                        # Assign this date as the location
                            data.loc[index, "isHoliday"] = 1
                            occurence_match=1
                        else:
                            occurence +=1

    return data




# %%
holidays = [
    {
        "holiday_name": "New Year's Day",
        "month": 1,
        "date" : 1,
        "occurence": 1,
        "startYear": 1870, # When it was recognized as a US federal holiday

    },
    {
        "holiday_name": "Juneteenth", # Only applicable after 2022 
        "month": 6,
        "date" : 19,
        "occurence": 1,
        "startYear":2021,
        

    },
    {
        "holiday_name": "Independence Day",
        "month": 7,
        "date" : 4,
        "occurence": 1,
        "startYear": 1870,

    },
    {
        "holiday_name": "Veterans Day",
        "month": 11,
        "date" : 11,
        "occurence": 1,
        "startYear": 1954,

    },
    {
        "holiday_name": "Christmas Day",
        "month": 12,
        "date" : 25,
        "occurence": 1, 
        "startYear": 1870,

    },
    {
        "holiday_name": "Martin Luther King Jr",
        "month": 1,
        "date" : "monday",
        "occurence": 3,# Third monday in janurary
        "startYear":1986,

    },
    {
        "holiday_name": "Memorial Day",
        "month": 2,
        "date" : "february",
        "occurence": 5, # Last monday of the month, could be 5 or 6 idk 
        "startYear": 1971,

    },
    {
        "holiday_name": "Labor Day",
        "month": 9,
        "date" : "monday",
        "occurence": 1,
        "startYear": 1894,

    },
    {
        "holiday_name": "Columbus Day",
        "month": 10,
        "date" : "monday",
        "occurence": 2,
        "startYear":1937,

    },
    {
        "holiday_name": "Thanksgiving Day",
        "month": 11,
        "date" : "thursday",
        "occurence": 4,
        "startYear":1870,

    }

]

# %%
def combineDataFrames(lmpData : pd.DataFrame, weatherData: pd.DataFrame) -> pd.DataFrame:
    
    combinedData = pd.DataFrame(columns=list(lmpData.columns )+ list(weatherData.columns))
    lmpData = lmpData.reset_index(drop=True)
    weatherData = weatherData.reset_index(drop=True)

    for index, lmpRow in lmpData.iterrows():
        weatherRow = weatherData.iloc[index]

        lmpYear = int(lmpRow['datetime_beginning_utc'].split(" ")[0].split("/")[2])
        lmpMonth = int(lmpRow['datetime_beginning_utc'].split(" ")[0].split("/")[0])
        lmpDay = int(lmpRow['datetime_beginning_utc'].split(" ")[0].split("/")[1])

        weatherYear = int(weatherRow['time'].split("T")[0].split("-")[0])
        weatherMonth = int(weatherRow ['time'].split("T")[0].split("-")[1])
        weatherDay = int(weatherRow ['time'].split("T")[0].split("-")[2])


        # IF CONDITIONS MET ****

        if lmpYear == weatherYear and lmpMonth == weatherMonth and lmpDay == weatherDay:

            currentCombination = pd.DataFrame([pd.concat([lmpRow.drop(['hourTime']),weatherRow],axis=0)], columns=list(lmpRow.index) + list(weatherRow.index))

            combinedData = pd.concat([combinedData, currentCombination], axis=0)

        else:
            print(f"The Index: {index}\nWeather Data: Year: {weatherYear}, Month: {weatherMonth}, Day: {weatherDay}\nLmp Data: Year: {lmpYear}, Month: {lmpMonth}, Day: {lmpDay}")
            break
    

    return combinedData



        


# %%


# %%
# modified total lmp delta of relative deviation |(a - r) / a|
def total_lmp_delta(data: pd.DataFrame) -> None:
    data['total_lmp_delta'] = 0
    for i in list(data.index):
        a = data.loc[i, "total_lmp_da"]
        r = data.loc[i, "total_lmp_rt"]
        # Compute relative deviation: |(a - r) / a|, handle division by zero
        if abs(a) < 1e-6:  # If a is close to zero, use a small constant
            a = 1e-6
        data.loc[i, "total_lmp_delta"] = abs((a - r) / a)
# %%
def addTarget(df: pd.DataFrame) -> None:
    if 'total_lmp_delta' not in df.columns:
        raise ValueError("'total_lmp_delta' column is missing")
    if df['total_lmp_delta'].isna().any():
        raise ValueError("NaN values found in 'total_lmp_delta'")
    
    num = len(df['total_lmp_delta'])
    top_5_percent_index = num * 0.95
    max_5 = sorted(df['total_lmp_delta'])[int(top_5_percent_index)]
    
    # Vectorized assignment
    df['target_c'] = (df['total_lmp_delta'] >= max_5).astype(int)
    
    df['target_c'] = df['target_c'].shift(-1)
    df['target_r'] = df['total_lmp_delta'].shift(-1)
    df.dropna(inplace=True)
    
    print("After addTarget, target_c value counts:\n", df['target_c'].value_counts())
    print("After addTarget, columns:", df.columns.tolist())
# %%
# Do 8AM and 8PM

# %%
6/13/2022

# %%
12/24/2022 

# %%
com = pd.read_pickle("hourlyLmpData/westernData/lmp_data_8PM.csv")
for index, row in com.iterrows():
    if row['total_lmp_da'] - row['total_lmp_rt'] < -1000:
        print(f"{index}\n{row}")

# %%
total_lmp_delta(com)
addTarget(com)
com[["datetime_beginning_utc","total_lmp_delta","target_r"]]

# %%


# %%
from sklearn.svm import SVR, SVC
from sklearn.linear_model import LinearRegression, Ridge, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.neighbors import KNeighborsClassifier

# %%
# Add confusion matrix to results 
from sklearn.metrics import confusion_matrix, mean_squared_error

# %%
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


# %%
def plotLmpData():
    PATH = "hourlyLmpData/westernData"
    lmpDirs = os.listdir(PATH)
    num = len(lmpDirs)
    row = 6
    col = 4
    f, axs = plt.subplots(2,2,figsize=(30,30))
    for index in range(1,num+1,1):
        dir = lmpDirs[index-1]

        lmpData = pd.read_pickle(PATH + "/" + dir)

        dataTime = dir.split("_")[2]

        plt.subplot(row, col, index)
        plt.plot(lmpData['total_lmp_da'] - lmpData['total_lmp_rt'])
        plt.xlabel("Day")
        plt.ylabel("Lmp Delta")
        plt.title(f"LMP DELTA AT {dataTime}")

        
    
    plt.show()

# %%
plotLmpData()

# %%
import tensorflow as tf
print("TensorFlow version:", tf.__version__)


# %%
#'lmp_data_10AM.csv
def trainingResults(LMP_PATH : str, WEATHER_PATH : str,c_models: list, r_models : list) -> dict:
    results = {}

    for items in os.listdir(LMP_PATH):
        lmp = pd.read_pickle(LMP_PATH + "/" + items)

        hold = items.split("_")
        hold[0] = "weather"

        dataTime = hold[2]

        hold = "_".join(hold)

        # Results time
        results[dataTime] = {}



        weather = pd.read_pickle(WEATHER_PATH + "/" + hold)


        com = combineDataFrames(lmp, weather)
        
        
        applyHoliday(com, holidays)
        total_lmp_delta(com)
        addTarget(com)

        
 
        inputs = (com.describe().columns[:-2])
        outputs = (com.describe().columns[-2:])

        scaler = StandardScaler()
        scaler.fit(com[inputs])
        X = scaler.transform(com[inputs])


        X_train, X_test, y_train, y_test = train_test_split(X,com[outputs]["target_c"],test_size=0.2, shuffle=False)
        X_train_r, X_test_r, y_train_r, y_test_r = train_test_split(X, com[outputs]['target_r'], test_size=0.2, shuffle=False)


        print(f"\n************************** DATE : {dataTime} **************************\n")
        print("***CLASSIFICATION***\n")
        for model in c_models:
            
            print(f"{str(model)}")
            results[dataTime][str(model)] = {}

            model.fit(X_train, y_train)

            trainResult = model.score(X_train,y_train)
            testResult = model.score(X_test,y_test)
            confusionMatrix = confusion_matrix(y_test, model.predict(X_test))

            print("Training", trainResult)
            print("Testing: ",testResult)
            print()
            print(confusionMatrix)
            print()

            results[dataTime][str(model)]["training"] =  trainResult
            results[dataTime][str(model)]["testing"] = testResult
            results[dataTime][str(model)]["confusion_matrix"] = confusionMatrix

        
        print("\n***REGRESSION***\n")
        for model in r_models:
            print(f"{str(model)}")
            results[dataTime][str(model)] = {}

            model.fit(X_train_r, y_train_r)
            
            trainResult = model.score(X_train,y_train)
            testResult = model.score(X_test,y_test)
            meanSqauredError = mean_squared_error(y_test_r, model.predict(X_test_r))

            print("Training",trainResult)
            print("Testing",testResult)
            print()
            print(meanSqauredError)
            print()

            results[dataTime][str(model)]["training"] =  trainResult
            results[dataTime][str(model)]["testing"] = testResult
            results[dataTime][str(model)]["meanSqauredError"] = meanSqauredError
        

        print("\n***Neural Network***\n")
        model = tf.keras.models.Sequential([
        tf.keras.layers.Embedding(input_dim=X_train.shape[0],output_dim=64),
        tf.keras.layers.LSTM(128, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation='relu'),
        #tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(3) # 0, 1, 2 
        ])

        results[dataTime]["NeuralNetwork"] = {}

        loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True)


        model.compile(optimizer='adam',
                    loss=loss_fn,
                    metrics=['accuracy'],
                    )

        print(model)

        history = model.fit(X_train, y_train, epochs=50, verbose=0)
        test_results = model.evaluate(X_test, y_test, verbose=2)
        print(f"accuracy: {test_results[1]} - loss: {test_results[0]}")

        results[dataTime]["NeuralNetwork"]["loss"] =  test_results[0]
        results[dataTime]["NeuralNetwork"]["testing"] = test_results[1]



    return results

        







        



# %%
'''
Neg: Negative: 0 : Not an outlier
Neu: Neutral : 1 : Postive outlier
Pos: Positive : 2 : Negatve outlier 

                    Predicted
                Neg     Neu     Pos
                
        Neg    NegNeg  NegNeu  NegPos
Actual  Neu    NeuNeg  NeuNeu  NeuPos
        Pos    PosNeg  PosNeu  PosPos

        

        # 
'''

# %%
'''result = trainingResults("hourlyLmpData/westernData", "hourlyWeatherData/openMeteo",
 [LogisticRegression(), RandomForestClassifier(), DecisionTreeClassifier(), SVC(), KNeighborsClassifier(n_neighbors=3)],
 []#Ridge(), SVR(), DecisionTreeRegressor(), LinearRegression(), RandomForestRegressor()]# 
 ) '''

# %%
# One page summary of results
# Put it on github 


# %%
def bestTestResult(result : dict) -> None:
    hours = [hour for hour in result]
    models = [model for model in result[hours[0]]]


    bestResults = {
        "max_test_acc": 0,
        "max_test_confusion_matrix": 0,
        "min_loss": 2,
        "best_model": "",
        "best_hour": "",
    }

    for hour in hours:

        for model in models:

            if result[hour][model].get("testing") and result[hour][model].get("testing") > bestResults["max_test_acc"]:
                bestResults["max_test_acc"] = result[hour][model].get("testing")
                bestResults["best_hour"] = hour
                bestResults["best_model"] = model


                # Check for matrix. If not matrix, then either means squared error or loss
                if result[hour][model].get("confusion_matrix") is not None:
                    bestResults['max_test_confusion_matrix'] = result[hour][model]['confusion_matrix']
                
                if result[hour][model].get("loss"):
                    bestResults['min_loss'] = result[hour][model]['loss']
                else:
                    bestResults['min_loss'] = ""
    

    return bestResults



# %%
def loadDataByHour(hour : str) -> pd.DataFrame:

    lmpPath = f"hourlyLmpData/westernData/lmp_data_{hour}.csv"
    weatherPath = f"hourlyWeatherData/openMeteo/weather_data_{hour}.csv"

    lmp = pd.read_pickle(lmpPath)
    weather = pd.read_pickle(weatherPath)


    com = combineDataFrames(lmp, weather)
    
    
    applyHoliday(com, holidays)
    total_lmp_delta(com)
    addTarget(com)

    return com

# %%


# %%
# 90% of data is not an outlier 
# 90% acc == 0%
# 95% acc = 50%
# 96% acc = 60%

# 97-98% = 70%-80%

# %%


# %%
# why regression worse than classification
# Why discrepancy between real time and day ahead (did they not know ahead)
# Regulation might have changed due to a cap (potential 1000)


# %%
data = loadDataByHour("8PM")

inputs = (data.describe().columns[:-2])
outputs = (data.describe().columns[-2:])

scaler = StandardScaler()
scaler.fit(data[inputs])
X = scaler.transform(data[inputs])

X_train, X_test, y_train, y_test = train_test_split(X, data[outputs]["target_c"], test_size=0.2, shuffle=False)

# %%
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# %%


# %%
from sklearn.model_selection import GridSearchCV
# Logistic Regression: default params is best 
# SVC : Default params but (C=0.01)
# RandomForestClassifier: {'criterion': 'log_loss', 'min_samples_leaf': 8, 'n_estimators': 10}
# Neural Network : Embedding, LSTM(128), Dropout(0.2), Dense(10), Dropout(0.2), Dense(3)
# xgboostClassifier: params = {'colsample_bytree': 0.6, 'gamma': 5, 'max_depth': 3, 'min_child_weight': 5, 'subsample': 0.8}

# LogisticRegression, RandomForestClassifier, SVC, NeuralNetwork,
# KNeighborsClassifier, Xgboost, LightGBM, 

# %%
from sklearn.metrics import classification_report

def evaluate_anomaly_detection(y_true, y_pred, model_name):
    print(f"{model_name} detection results:")
    print(classification_report(y_true, y_pred, target_names=['normal', 'large relative deviation'], zero_division=0))

# %%
# Load data
com = loadDataByHour("8PM")
total_lmp_delta(com)
addTarget(com)
#add lagged features
com['lagged_lmp_da'] = com['total_lmp_da'].shift(1)
com['lagged_lmp_rt'] = com['total_lmp_rt'].shift(1)
com['lagged_delta'] = com['total_lmp_delta'].shift(1)
com.dropna(inplace=True)

com = applyHoliday(com, holidays)
print("After applyHoliday, columns:", com.columns.tolist())
# %%
inputs = [
    'lagged_lmp_da', 'lagged_delta',
    'apparent_temperature (°C)', 'wind_gusts_10m (km/h)', 'pressure_msl (hPa)',
    'soil_temperature_0_to_7cm (°C)', 'soil_moisture_0_to_7cm (m³/m³)',
    'isHoliday'
] 

# add hour one-hot encoding
inputs += [col for col in com.columns if col.startswith('hour_')]

scaler = StandardScaler()
scaler.fit(com[inputs])
X = scaler.transform(com[inputs])
y = com['target_c']
outputs = com.describe().columns[-2:]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, com['target_c'], test_size=0.2, shuffle=True, stratify=y, random_state=42)

# Print original data distribution
print("\nOriginal data distribution:")
for i in range(3):
    print(f"Class {i}: {np.sum(y_train == i)} samples")

# %%
# RandomForest
rlf_params = {
    'criterion': 'entropy',
    'min_samples_leaf': 2,
    'n_estimators': 300,
    'max_depth': 20,
    'class_weight': 'balanced',
    'min_samples_split': 5
}
rf = RandomForestClassifier(**rlf_params)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
print("\nRandom Forest evaluation results:")
evaluate_anomaly_detection(y_test, y_pred, "Random Forest")

# %%
# XGBoost
xgb_params = {
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'max_depth': 10,
    'min_child_weight': 1,
    'subsample': 0.8,
    'scale_pos_weight': 50,
    'learning_rate': 0.05,
    'n_estimators': 300
}
xgb = XGBClassifier(**xgb_params)
xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
print("\nXGBoost evaluation results:")
evaluate_anomaly_detection(y_test, y_pred, "XGBoost")

# %%
# SVC
svc_params = {
    'C': 1.0,
    'probability': True,
    'class_weight': 'balanced',
    'kernel': 'rbf',
    'gamma': 'scale',
    'cache_size': 1000
}
svc = SVC(**svc_params)
svc.fit(X_train, y_train)
y_pred = svc.predict(X_test)
print("\nSVC evaluation results:")
evaluate_anomaly_detection(y_test, y_pred, "SVC")

# %%
# Logistic Regression
param_grid = {
    'C': [0.1, 0.5, 1, 5, 10],
    'class_weight': ['balanced', {0:1, 1:2}, {0:1, 1:4}, {0:1, 1:8}]
}
lr = LogisticRegression(max_iter=2000, solver='saga', tol=1e-4)
grid = GridSearchCV(lr, param_grid, scoring='f1_macro', cv=5)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
best_lr = grid.best_estimator_
y_pred = best_lr.predict(X_test)
print("\nLogistic Regression evaluation results:")
evaluate_anomaly_detection(y_test, y_pred, "Logistic Regression")

# %%
# Print test set distribution
print("\nTest set distribution:")
for i in range(3):
    print(f"Class {i}: {np.sum(y_test == i)} samples")

# %%
# 5.19.2025 Use all hours data to train and evaluate model
import os

# 1. merge all data from all hours
all_coms = []
LMP_PATH = "hourlyLmpData/westernData"
WEATHER_PATH = "hourlyWeatherData/openMeteo"
for items in os.listdir(LMP_PATH):
    lmp = pd.read_pickle(os.path.join(LMP_PATH, items))
    hold = items.split("_")
    hold[0] = "weather"
    weather_file = "_".join(hold)
    weather = pd.read_pickle(os.path.join(WEATHER_PATH, weather_file))
    hour = hold[2].replace(".csv", "")
    com = combineDataFrames(lmp, weather)
    com = applyHoliday(com, holidays)
    com['hour'] = hour
    total_lmp_delta(com)
    addTarget(com)
    com['lagged_lmp_da'] = com['total_lmp_da'].shift(1)
    com['lagged_lmp_rt'] = com['total_lmp_rt'].shift(1)
    com['lagged_delta'] = com['total_lmp_delta'].shift(1)
    com.dropna(inplace=True)
    all_coms.append(com)


# 2. merge all data from all hours
all_data = pd.concat(all_coms, axis=0, ignore_index=True)
all_data = pd.get_dummies(all_data, columns=['hour']) #one-hot encoding hour
# 3. features and labels
k = 3  # try small k first

for i in range(1, k+1):
    all_data[f'total_lmp_da_lag_{i}'] = all_data['total_lmp_da'].shift(i)

all_data = all_data.dropna().reset_index(drop=True)

inputs = [
    'lagged_lmp_da', 'lagged_delta',
    'apparent_temperature (°C)', 'wind_gusts_10m (km/h)', 'pressure_msl (hPa)',
    'soil_temperature_0_to_7cm (°C)', 'soil_moisture_0_to_7cm (m³/m³)',
    'isHoliday'
] + [f'total_lmp_da_lag_{i}' for i in range(1, k+1)]

# add hour one-hot encoding
inputs += [col for col in all_data.columns if col.startswith('hour_')]

scaler = StandardScaler()
scaler.fit(all_data[inputs])
X = scaler.transform(all_data[inputs])
y = all_data['target_c']

# 4. split data into  testing
from sklearn.model_selection import train_test_split
X_temp, X_test, y_temp, y_test = train_test_split(
    X, y, test_size=0.2, shuffle=True, stratify=y, random_state=42
)

# validating set
X_train, X_val, y_train, y_val = train_test_split(
    X_temp, y_temp, test_size=0.2, shuffle=True, stratify=y_temp, random_state=42
)

# %%
from sklearn.metrics import precision_recall_curve
def find_optimal_threshold(y_test, y_pred_proba):
    # 1. calculate precision and recall at different thresholds
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba[:, 1])
    
    # 2. calculate F1
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    
    # 3. Find threshold by optimal F1
    optimal_threshold = thresholds[np.argmax(f1_scores)]
    
    # 4. Visulization
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], label='Precision')
    plt.plot(thresholds, recalls[:-1], label='Recall')
    plt.plot(thresholds, f1_scores[:-1], label='F1-score')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal threshold: {optimal_threshold:.2f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Metrics vs. Threshold')
    plt.legend()
    plt.grid(True)
    plt.show()
    
    return optimal_threshold
# %%
# train and evaluate model (RF)
rf = RandomForestClassifier(
    criterion='entropy',
    min_samples_leaf=2,
    n_estimators=300,
    max_depth=20,
    class_weight='balanced',
    min_samples_split=5
)
rf.fit(X_train, y_train)
#  use validaion set to calculate optimal
y_val_proba = rf.predict_proba(X_val)
optimal_threshold_rf = find_optimal_threshold(y_val, y_val_proba)
# use optimal threshold evaluate on testing set
y_test_proba = rf.predict_proba(X_test)
y_pred_optimal = (y_test_proba[:, 1] >= optimal_threshold).astype(int)
print("\nResult of optimal threshold")
evaluate_anomaly_detection(y_test, y_pred_optimal, "Random Forest")

# %%
import matplotlib.pyplot as plt
import numpy as np

importances = rf.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = np.array(inputs)

plt.figure(figsize=(12, 6))
plt.barh(feature_names[indices], importances[indices], color='maroon')
plt.xlabel("Feature Importance")
plt.title("Random Forest Feature Importances")
plt.gca().invert_yaxis()
plt.show()

# print top 10 features
for i in indices[:10]:
    print(f"{feature_names[i]}: {importances[i]:.4f}")


# %%
# train and evaluate model (XGBoost)
xgb_params = {
    'colsample_bytree': 0.8,
    'gamma': 0.1,
    'max_depth': 10,
    'min_child_weight': 1,
    'subsample': 0.8,
    'scale_pos_weight': 50,
    'learning_rate': 0.05,
    'n_estimators': 300
}
xgb = XGBClassifier(**xgb_params)
xgb.fit(X_train, y_train)
# use validation set to calculate optimal
y_val_proba = xgb.predict_proba(X_val)
optimal_threshold_xgb = find_optimal_threshold(y_val, y_val_proba)
# use optimal threshold evaluate on testing set
y_test_proba = xgb.predict_proba(X_test)
y_pred_optimal = (y_test_proba[:, 1] >= optimal_threshold_xgb).astype(int)
print("\nXGBoost with optimal threshold results:")
evaluate_anomaly_detection(y_test, y_pred_optimal, "XGBoost")

# %%
import matplotlib.pyplot as plt
import numpy as np

# use randomForest
importances = xgb.feature_importances_
indices = np.argsort(importances)[::-1]
feature_names = np.array(inputs)

plt.figure(figsize=(12, 6))
plt.barh(feature_names[indices], importances[indices], color='maroon')
plt.xlabel("Feature Importance")
plt.title("XGBoost Feature Importances")
plt.gca().invert_yaxis()
plt.show()

# print first 10 feature
for i in indices[:10]:
    print(f"{feature_names[i]}: {importances[i]:.4f}")
# %%
# train and evaluate model (SVC)
svc_params = {
    'C': 1.0,
    'probability': True,
    'class_weight': 'balanced',
    'kernel': 'rbf',
    'gamma': 'scale',
    'cache_size': 1000
}
svc = SVC(**svc_params)
svc.fit(X_train, y_train)
# use validation set to calculate optimal
y_val_proba = svc.predict_proba(X_val)
optimal_threshold_svc = find_optimal_threshold(y_val, y_val_proba)
# use optimal threshold evaluate on testing set
y_test_proba = svc.predict_proba(X_test)
y_pred_optimal = (y_test_proba[:, 1] >= optimal_threshold_svc).astype(int)
print("\nSVC with optimal threshold results:")
evaluate_anomaly_detection(y_test, y_pred_optimal, "SVC")

# %%
# train and evaluate model (Logistic Regression)
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 0.5, 1, 5, 10],
    'class_weight': ['balanced', {0:1, 1:2}, {0:1, 1:4}, {0:1, 1:8}]
}
lr = LogisticRegression(max_iter=2000, solver='saga', tol=1e-4)
grid = GridSearchCV(lr, param_grid, scoring='f1_macro', cv=5)
grid.fit(X_train, y_train)
print("Best params:", grid.best_params_)
best_lr = grid.best_estimator_
# use validation set to calculate optimal
y_val_proba = best_lr.predict_proba(X_val)
optimal_threshold_lr = find_optimal_threshold(y_val, y_val_proba)
# use optimal threshold evaluate on testing set
y_test_proba = best_lr.predict_proba(X_test)
y_pred_optimal = (y_test_proba[:, 1] >= optimal_threshold_lr).astype(int)
print("\nLogistic Regression with optimal threshold results:")
evaluate_anomaly_detection(y_test, y_pred_optimal, "Logistic Regression")

# %%
# Print all optimal thresholds
print("\nOptimal thresholds for each model:")
print(f"Random Forest: {optimal_threshold_rf:.4f}")
print(f"XGBoost: {optimal_threshold_xgb:.4f}")
print(f"SVC: {optimal_threshold_svc:.4f}")
print(f"Logistic Regression: {optimal_threshold_lr:.4f}")

# %%
from sklearn.ensemble import VotingClassifier

# build ensemble model
ensemble = VotingClassifier(
    estimators=[
        ('rf', rf),
        ('xgb', xgb),
        ('svc', svc),
        ('lr', best_lr)
    ],
    voting='soft'  # 'soft' means use probability average, 'hard' means use class voting
)

# train ensemble model
ensemble.fit(X_train, y_train)

# find optimal threshold on validation set
y_val_proba = ensemble.predict_proba(X_val)
optimal_threshold_ensemble = find_optimal_threshold(y_val, y_val_proba)

# evaluate on testing set
y_test_proba = ensemble.predict_proba(X_test)
y_pred_optimal = (y_test_proba[:, 1] >= optimal_threshold_ensemble).astype(int)
print("\nEnsemble with optimal threshold results:")
evaluate_anomaly_detection(y_test, y_pred_optimal, "Ensemble")
# %%
