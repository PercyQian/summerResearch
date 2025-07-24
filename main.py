# %%
import pandas as pd 
import matplotlib.pyplot as plt 
# %matplotlib inline 
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
# Push to repo 




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
#old version of addTarget based on 95th percentile of total_lmp_delta, not using anymore
'''
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
'''

def addTarget_corrected(df: pd.DataFrame, k=2.5) -> None:
    """
    Use historical baseline relative difference method to define anomalies 
    
    Parameters:
    - df: DataFrame containing total_lmp_delta
    - k: standard deviation multiple threshold (default 2.5)
    """
    if 'total_lmp_delta' not in df.columns:
        raise ValueError("'total_lmp_delta' column is missing")
    if df['total_lmp_delta'].isna().any():
        raise ValueError("NaN values found in 'total_lmp_delta'")
    
    print(f"=== use historical baseline relative difference method to define anomalies (k={k}) ===")
    
    # calculate error (here total_lmp_delta is the difference between RT-DA)
    df['error'] = df['total_lmp_delta']
    
    # extract time features
    try:
        if 'datetime_beginning_utc' in df.columns:
            if df['datetime_beginning_utc'].dtype == 'object':
                df['datetime_beginning_utc'] = pd.to_datetime(df['datetime_beginning_utc'])
            
            df['hour_num'] = df['datetime_beginning_utc'].dt.hour
            df['dow'] = df['datetime_beginning_utc'].dt.dayofweek
        else:
            # if no time column, try to extract from hourTime
            if 'hourTime' in df.columns:
                hour_mapping = {
                    '12AM': 0, '1AM': 1, '2AM': 2, '3AM': 3, '4AM': 4, '5AM': 5, '6AM': 6, '7AM': 7, '8AM': 8, '9AM': 9, '10AM': 10, '11AM': 11,
                    '12PM': 12, '1PM': 13, '2PM': 14, '3PM': 15, '4PM': 16, '5PM': 17, '6PM': 18, '7PM': 19, '8PM': 20, '9PM': 21, '10PM': 22, '11PM': 23
                }
                df['hour_num'] = df['hourTime'].map(hour_mapping).fillna(12).astype(int)
                df['dow'] = 1  # default is Tuesday
            else:
                # if none, use default value
                df['hour_num'] = 12  # default is 12PM
                df['dow'] = 1  # default is Tuesday
    except Exception as e:
        print(f"⚠️ time feature extraction failed: {e}")
        df['hour_num'] = 12
        df['dow'] = 1
    
    # calculate statistical baseline for each time period
    print("calculate statistical baseline for each time period...")
    baseline_stats = df.groupby(['hour_num', 'dow'])['error'].agg(
        mu='mean',
        sigma='std', 
        count='count'
    ).reset_index()
    
    # handle cases where standard deviation is NaN or 0
    global_sigma = df['error'].std()
    if pd.isna(global_sigma) or global_sigma == 0:
        global_sigma = 1.0  # default standard deviation
    
    baseline_stats['sigma'] = baseline_stats['sigma'].fillna(global_sigma)
    baseline_stats.loc[baseline_stats['sigma'] == 0, 'sigma'] = global_sigma
    
    print(f"established {len(baseline_stats)} time period baselines")
    
    # merge baseline statistics
    merged_df = df.merge(baseline_stats, on=['hour_num', 'dow'], how='left')
    
    # for time periods without historical baseline, use global statistics
    global_mu = merged_df['error'].mean()
    merged_df['mu'] = merged_df['mu'].fillna(global_mu)
    merged_df['sigma'] = merged_df['sigma'].fillna(global_sigma)
    
    # calculate
    merged_df['abs_deviation'] = np.abs(merged_df['error'] - merged_df['mu'])
    merged_df['threshold'] = k * merged_df['sigma']
    merged_df['is_anomaly'] = (merged_df['abs_deviation'] > merged_df['threshold']).astype(int)
    
    # add results to original DataFrame
    df['target_c'] = merged_df['is_anomaly'].astype(int)
    df['target_r'] = df['total_lmp_delta']
    
    # statistics result
    total_anomalies = df['target_c'].sum()
    anomaly_rate = total_anomalies / len(df) * 100
    
    print(f"anomaly detection result:")
    print(f"total anomalies: {total_anomalies}")
    print(f"anomaly rate: {anomaly_rate:.2f}%")
    print("After addTarget_corrected, target_c value counts:\n", df['target_c'].value_counts())
    
    # clean temporary columns (only clean temporary columns in original df)
    cols_to_drop = ['error', 'hour_num', 'dow']
    for col in cols_to_drop:
        if col in df.columns:
            df.drop(col, axis=1, inplace=True)

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
addTarget_corrected(com)
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
        addTarget_corrected(com)

        
 
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
    addTarget_corrected(com)

    return com

# %%


# %%
# 90% of data is not an outlier 
# 90% acc == 0%
# 95% acc = 50%
# 96% acc = 60%

# 97-98% = 70%-80%


# %%
# why regression worse than classification
# Why discrepancy between real time and day ahead (did they not know ahead)
# Regulation might have changed due to a cap (potential 1000)



# %%
'''
data = loadDataByHour("8PM")

inputs = (data.describe().columns[:-2])
outputs = (data.describe().columns[-2:])

scaler = StandardScaler()
scaler.fit(data[inputs])
X = scaler.transform(data[inputs])

X_train, X_test, y_train, y_test = train_test_split(X, data[outputs]["target_c"], test_size=0.2, shuffle=False)








from sklearn.model_selection import GridSearchCV
# Logistic Regression: default params is best 
# SVC : Default params but (C=0.01)
# RandomForestClassifier: {'criterion': 'log_loss', 'min_samples_leaf': 8, 'n_estimators': 10}
# Neural Network : Embedding, LSTM(128), Dropout(0.2), Dense(10), Dropout(0.2), Dense(3)
# xgboostClassifier: params = {'colsample_bytree': 0.6, 'gamma': 5, 'max_depth': 3, 'min_child_weight': 5, 'subsample': 0.8}

# LogisticRegression, RandomForestClassifier, SVC, NeuralNetwork,
# KNeighborsClassifier, Xgboost, LightGBM, 



# Load data
com = loadDataByHour("8PM")
total_lmp_delta(com)
addTarget_corrected(com)
#add lagged features
com['lagged_lmp_da'] = com['total_lmp_da'].shift(1)
    # 移除信息泄漏特征：lagged_lmp_rt 和 lagged_delta
    # 因为预测时无法获得上一小时的RT数据
com.dropna(inplace=True)

com = applyHoliday(com, holidays)
print("After applyHoliday, columns:", com.columns.tolist())

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


# Print test set distribution
print("\nTest set distribution:")
for i in range(3):
    print(f"Class {i}: {np.sum(y_test == i)} samples")
'''
#%%
from sklearn.metrics import classification_report

def evaluate_anomaly_detection(y_true, y_pred, model_name):
    print(f"{model_name} detection results:")
    report = classification_report(y_true, y_pred, target_names=['normal', 'large relative deviation'], zero_division=0, output_dict=True)
    print(classification_report(y_true, y_pred, target_names=['normal', 'large relative deviation'], zero_division=0))
    
    # return F1 score of anomaly class
    try:
        anomaly_f1 = report['large relative deviation']['f1-score']
        print(f"📊 F1 score of anomaly class: {anomaly_f1:.4f}")
        return anomaly_f1
    except KeyError as e:
        print(f"⚠️ cannot get F1 score of anomaly class: {e}")
        print(f"available keys: {list(report.keys())}")
        # try to use index 1 (anomaly class)
        try:
            # if target_names is not correctly mapped, try to use index 1 directly
            if '1' in report:
                anomaly_f1 = report['1']['f1-score']
                print(f"📊 use index 1 to get F1 score of anomaly class: {anomaly_f1:.4f}")
                return anomaly_f1
            else:
                print("❌ cannot find F1 score of anomaly class, return 0.0")
                return 0.0
        except:
            print("❌ cannot find F1 score of anomaly class, return 0.0")
            return 0.0

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
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
    addTarget_corrected(com)
    com['lagged_lmp_da'] = com['total_lmp_da'].shift(1)
    # remove information leakage features: lagged_lmp_rt and lagged_delta
    # because RT data is not available at prediction time
    com.dropna(inplace=True)
    all_coms.append(com)


# 2. merge all data from all hours
all_data = pd.concat(all_coms, axis=0, ignore_index=True)
all_data = pd.get_dummies(all_data, columns=['hour']) #one-hot encoding hour
# 3. features and labels
k = 7  # try small k first

for i in range(1, k+1):
    all_data[f'total_lmp_da_lag_{i}'] = all_data['total_lmp_da'].shift(i)

all_data = all_data.dropna().reset_index(drop=True)

inputs = [
    'lagged_lmp_da',  # only keep lagged DA features, because DA data is available in advance
    'apparent_temperature (°C)', 'wind_gusts_10m (km/h)', 'pressure_msl (hPa)',
    'soil_temperature_0_to_7cm (°C)', 'soil_moisture_0_to_7cm (m³/m³)',
    'isHoliday'
] + [f'total_lmp_da_lag_{i}' for i in range(1, k+1)]  # multiple lagged DA features

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
from sklearn.metrics import precision_recall_curve, f1_score
def find_optimal_threshold(y_test, y_pred_proba):
    # 1. calculate precision and recall at different thresholds for anomaly class (class 1)
    precisions, recalls, thresholds = precision_recall_curve(y_test, y_pred_proba)
    
    # 2. calculate F1 for anomaly class (class 1)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls)
    # handle division by zero
    f1_scores = np.nan_to_num(f1_scores)
    
    # 3. Find threshold by optimal F1 for anomaly class
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    print(f"optimal threshold: {optimal_threshold:.4f}")
    print(f"anomaly class optimal F1: {f1_scores[optimal_idx]:.4f}")
    print(f"corresponding precision: {precisions[optimal_idx]:.4f}")
    print(f"corresponding recall: {recalls[optimal_idx]:.4f}")
    
    # 4. Visulization
    plt.figure(figsize=(10, 6))
    plt.plot(thresholds, precisions[:-1], label='Precision (Anomaly Class)')
    plt.plot(thresholds, recalls[:-1], label='Recall (Anomaly Class)')
    plt.plot(thresholds, f1_scores[:-1], label='F1-score (Anomaly Class)')
    plt.axvline(x=optimal_threshold, color='r', linestyle='--', label=f'Optimal threshold: {optimal_threshold:.4f}')
    plt.xlabel('Threshold')
    plt.ylabel('Score')
    plt.title('Anomaly Class Metrics vs. Threshold')
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
y_val_proba = rf.predict_proba(X_val)[:, 1]
print("\n=== Random Forest threshold optimization ===")
optimal_threshold_rf = find_optimal_threshold(y_val, y_val_proba)
# use optimal threshold evaluate on testing set
y_test_proba = rf.predict_proba(X_test)[:, 1]
y_pred_optimal = (y_test_proba >= optimal_threshold_rf).astype(int)
print("\nRandom Forest optimal threshold test results:")
rf_f1 = evaluate_anomaly_detection(y_test, y_pred_optimal, "Random Forest")

# save Random Forest model
models_dict_temp = {'Random Forest': rf}
optimal_thresholds_temp = {'Random Forest': optimal_threshold_rf}
f1_scores_temp = {'Random Forest': rf_f1}
save_models_if_better(models_dict_temp, scaler, optimal_thresholds_temp, f1_scores_temp)
print("✅ Random Forest model saved to trained_models/")

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
y_val_proba = xgb.predict_proba(X_val)[:, 1]
print("\n=== XGBoost threshold optimization ===")
optimal_threshold_xgb = find_optimal_threshold(y_val, y_val_proba)
# use optimal threshold evaluate on testing set
y_test_proba = xgb.predict_proba(X_test)[:, 1]
y_pred_optimal = (y_test_proba >= optimal_threshold_xgb).astype(int)
print("\nXGBoost optimal threshold test results:")
xgb_f1 = evaluate_anomaly_detection(y_test, y_pred_optimal, "XGBoost")

# save XGBoost model
models_dict_temp = {'XGBoost': xgb}
optimal_thresholds_temp = {'XGBoost': optimal_threshold_xgb}
f1_scores_temp = {'XGBoost': xgb_f1}
save_models_if_better(models_dict_temp, scaler, optimal_thresholds_temp, f1_scores_temp)
print("✅ XGBoost model saved to trained_models/")

# %%
import matplotlib.pyplot as plt
import numpy as np

# use xgboost
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
y_val_proba = svc.predict_proba(X_val)[:, 1]
print("\n=== SVC threshold optimization ===")
optimal_threshold_svc = find_optimal_threshold(y_val, y_val_proba)
# use optimal threshold evaluate on testing set
y_test_proba = svc.predict_proba(X_test)[:, 1]
y_pred_optimal = (y_test_proba >= optimal_threshold_svc).astype(int)
print("\nSVC optimal threshold test results:")
svc_f1 = evaluate_anomaly_detection(y_test, y_pred_optimal, "SVC")

# save SVC model
models_dict_temp = {'SVC': svc}
optimal_thresholds_temp = {'SVC': optimal_threshold_svc}
f1_scores_temp = {'SVC': svc_f1}
save_models_if_better(models_dict_temp, scaler, optimal_thresholds_temp, f1_scores_temp)
print("✅ SVC model saved to trained_models/")

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
y_val_proba = best_lr.predict_proba(X_val)[:, 1]
print("\n=== Logistic Regression threshold optimization ===")
optimal_threshold_lr = find_optimal_threshold(y_val, y_val_proba)
# use optimal threshold evaluate on testing set
y_test_proba = best_lr.predict_proba(X_test)[:, 1]
y_pred_optimal = (y_test_proba >= optimal_threshold_lr).astype(int)
print("\nLogistic Regression optimal threshold test results:")
lr_f1 = evaluate_anomaly_detection(y_test, y_pred_optimal, "Logistic Regression")

# save Logistic Regression model
models_dict_temp = {'Logistic Regression': best_lr}
optimal_thresholds_temp = {'Logistic Regression': optimal_threshold_lr}
f1_scores_temp = {'Logistic Regression': lr_f1}
save_models_if_better(models_dict_temp, scaler, optimal_thresholds_temp, f1_scores_temp)
print("✅ Logistic Regression model saved to trained_models/")

# %%
# Print all optimal thresholds
print("\n=== all models optimal thresholds ===")
print(f"Random Forest: {optimal_threshold_rf:.4f}")
print(f"XGBoost: {optimal_threshold_xgb:.4f}")
print(f"SVC: {optimal_threshold_svc:.4f}")
print(f"Logistic Regression: {optimal_threshold_lr:.4f}")

# save all models
print("\n=== save all trained models ===")
models_dict_final = {
    'Random Forest': rf,
    'XGBoost': xgb,
    'SVC': svc,
    'Logistic Regression': best_lr
}

optimal_thresholds_final = {
    'Random Forest': optimal_threshold_rf,
    'XGBoost': optimal_threshold_xgb,
    'SVC': optimal_threshold_svc,
    'Logistic Regression': optimal_threshold_lr
}

# 收集所有F1分数
f1_scores_final = {
    'Random Forest': rf_f1,
    'XGBoost': xgb_f1,
    'SVC': svc_f1,
    'Logistic Regression': lr_f1
}

save_models_if_better(models_dict_final, scaler, optimal_thresholds_final, f1_scores_final)
print("✅ all models saved to trained_models/ (using corrected version of addTarget_corrected)")

# %%
'''
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
'''
# %%

def plot_precision_recall_comparison(models_dict, X_test, y_test):
    """
    draw precision-recall curves comparison for multiple models
    models_dict: dictionary, contains model name and model object
    X_test: test data features
    y_test: test data labels
    """
    plt.figure(figsize=(10, 8))
    
    # calculate and draw precision-recall curves for each model
    for model_name, model in models_dict.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
        
        # calculate F1 score
        f1 = 2 * (precision * recall) / (precision + recall + 1e-10)
        
        # draw precision-recall curve
        plt.plot(recall, precision, label=f'{model_name}', linewidth=2)
    
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curves Comparison')
    plt.legend(loc='best')
    plt.grid(True)
    plt.show()

# example
# models_dict = {
#     'Random Forest': rf,
#     'XGBoost': xgb,
#     'SVC': svc,
#     'Logistic Regression': best_lr
# }
# plot_precision_recall_comparison(models_dict, X_test, y_test)

# %%

# create model dictionary and draw comparison plot
models_dict = {
    'Random Forest': rf,
    'XGBoost': xgb,
    'SVC': svc,
    'Logistic Regression': best_lr
}


# %%
# %%
# Use Open-Meteo API to download weather data for Western Hub

import requests
import json
from datetime import datetime, timedelta
import time

zone_coords = {
    'AE':    (39.3643,  -74.4229),
    'BGE':   (39.2904,  -76.6122),
    'DPL':   (39.1582,  -75.5244),
    'JCPL':  (40.2171,  -74.7429),
    'METED': (40.3356,  -75.9269),
    'PECO':  (39.9526,  -75.1652),
    'PPL':   (40.6084,  -75.4902),
    'PEPCO': (38.9072,  -77.0369),
    'DOM':   (37.5407,  -77.4360),
    'APS':   (38.3498,  -81.6326),
    'PENELEC':(41.4089, -75.6624),
    'PSEG':  (40.7357,  -74.1724),
    'RECO':  (41.1486,  -73.9881),
}

def download_weather_data_from_meteo(start_date, end_date, latitude, longitude, location_name, save_path="weatherData/meteo/"):
    """
    Download weather data for Western Hub from Open-Meteo API
    
    Parameters:
    - start_date: start date (format: 'YYYY-MM-DD')
    - end_date: end date (format: 'YYYY-MM-DD')
    - save_path: save path
    """
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    
    # Open-Meteo API URL
    base_url = "https://archive-api.open-meteo.com/v1/archive"
    
    # request parameters
    params = {
        'latitude': latitude,
        'longitude': longitude,
        'start_date': start_date,
        'end_date': end_date,
        'hourly': [
            'apparent_temperature',
            'pressure_msl',
            'wind_gusts_10m',
            'soil_temperature_0_to_7cm',
            'soil_moisture_0_to_7cm'
        ],
        'timezone': 'America/Los_Angeles',
        'format': 'json'
    }
    
    print(f"Downloading weather data from Open-Meteo for {start_date} to {end_date}...")
    print(f"Location: {location_name} (latitude {latitude}, longitude {longitude})")
    
    try:
        # send API request
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        
        # parse JSON data
        weather_data = response.json()
        
        # convert to DataFrame
        df = pd.DataFrame()
        
        # add time column
        df['time'] = weather_data['hourly']['time']
        
        # add all weather variables
        for variable in params['hourly']:
            if variable in weather_data['hourly']:
                df[f'{variable} ({weather_data["hourly_units"][variable]})'] = weather_data['hourly'][variable]
        
        # save data
        filename = f"{save_path}{location_name}_weather_{start_date}_to_{end_date}.csv"
        df.to_csv(filename, index=False)
        
        # also save as pickle format
        pickle_filename = f"{save_path}{location_name}_weather_{start_date}_to_{end_date}.pkl"
        df.to_pickle(pickle_filename)
        
        print(f"Weather data saved to:")
        print(f"  CSV: {filename}")
        print(f"  Pickle: {pickle_filename}")
        print(f"Number of rows: {len(df)}")
        print(f"Number of columns: {len(df.columns)}")
        
        return df
        
    except requests.exceptions.RequestException as e:
        print(f"Error downloading weather data: {e}")
        return None
    except Exception as e:
        print(f"Error processing weather data: {e}")
        return None

def download_weather_for_all_zones(start_date, end_date, base_save_path="weatherData/meteo/"):
    """
    Download weather data for all zones for a given date range.
    
    Parameters:
    - start_date: start date (format: 'YYYY-MM-DD')
    - end_date: end date (format: 'YYYY-MM-DD')
    - base_save_path: base directory to save all zone data
    """
    for zone_name, (lat, lon) in zone_coords.items():
        print(f"\n--- Starting download for zone: {zone_name} ---")
        zone_save_path = os.path.join(base_save_path, zone_name)
        
        download_weather_data_from_meteo(
            start_date=start_date, 
            end_date=end_date, 
            latitude=lat, 
            longitude=lon, 
            location_name=zone_name, 
            save_path=zone_save_path
        )
        print(f"--- Finished download for zone: {zone_name} ---")

def process_meteo_weather_for_hourly_files(weather_data_path, output_path="hourlyWeatherData/meteo/"):
    """
    Process downloaded weather data into hourly files (match LMP data format)
    """
    import os
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    # read weather data
    if weather_data_path.endswith('.pkl'):
        weather_data = pd.read_pickle(weather_data_path)
    else:
        weather_data = pd.read_csv(weather_data_path)
    
    # add hour time column
    weather_data = addWeatherHourColumn(weather_data)
    
    # save by hour
    hours = weather_data['hourTime'].unique()
    
    for hour in hours:
        hour_data = weather_data[weather_data['hourTime'] == hour].reset_index(drop=True)
        filename = f"{output_path}weather_data_{hour}.pkl"
        hour_data.to_pickle(filename)
        print(f"Saved data for hour {hour}: {filename} ({len(hour_data)} records)")
    
    print(f"All hourly data saved to: {output_path}")

# %%
# Useage

# Download weather data for 2022-2025

download_weather_for_all_zones(start_date="2022-01-01", end_date="2025-06-06")


# %%
# save trained

import pickle
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

def save_trained_models(models_dict, scaler, optimal_thresholds, save_path="trained_models/"):
    """
    save trained models, scaler and optimal thresholds
    
    Parameters:
    - models_dict: trained models dictionary
    - scaler: trained scaler
    - optimal_thresholds: optimal thresholds for each model
    - save_path: save path
    """
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print("=== save trained models ===")
    
    # save models
    for model_name, model in models_dict.items():
        model_file = f"{save_path}{model_name.replace(' ', '_').lower()}_model.pkl"
        with open(model_file, 'wb') as f:
            pickle.dump(model, f)
        print(f"✅ {model_name} model saved: {model_file}")
    
    # save scaler
    scaler_file = f"{save_path}scaler.pkl"
    with open(scaler_file, 'wb') as f:
        pickle.dump(scaler, f)
    print(f"✅ scaler saved: {scaler_file}")
    
    # save optimal thresholds
    thresholds_file = f"{save_path}optimal_thresholds.pkl"
    with open(thresholds_file, 'wb') as f:
        pickle.dump(optimal_thresholds, f)
    print(f"✅ optimal thresholds saved: {thresholds_file}")
    
    print(f"all model components saved to: {save_path}")

def save_models_if_better(new_models_dict, new_scaler, new_optimal_thresholds, new_f1_scores, save_path="trained_models/"):
    """
    compare new models with existing models, only save if better
    
    Parameters:
    - new_models_dict: new trained models dictionary
    - new_scaler: new scaler
    - new_optimal_thresholds: new optimal thresholds
    - new_f1_scores: new model F1 scores dictionary {model_name: f1_score}
    - save_path: save path
    """
    import os
    from sklearn.metrics import f1_score
    
    print("=== compare model performance and save if better ===")
    
    # try to load existing models and F1 scores
    try:
        existing_models, existing_scaler, existing_thresholds = load_trained_models(save_path)
        has_existing_models = len(existing_models) > 0 and existing_scaler is not None
        
        # try to load existing F1 scores
        f1_scores_file = f"{save_path}model_f1_scores.pkl"
        if os.path.exists(f1_scores_file):
            with open(f1_scores_file, 'rb') as f:
                existing_f1_scores = pickle.load(f)
        else:
            existing_f1_scores = {}
    except:
        has_existing_models = False
        existing_models = {}
        existing_thresholds = {}
        existing_f1_scores = {}
    
    models_to_save = {}
    thresholds_to_save = {}
    f1_scores_to_save = {}
    
    for model_name, new_model in new_models_dict.items():
        # directly use F1 score from input
        new_f1 = new_f1_scores.get(model_name, 0.0)
        # handle None values
        if new_f1 is None:
            new_f1 = 0.0
            print(f"⚠️ F1 score of {model_name} is None, set to 0.0")
        
        new_threshold = new_optimal_thresholds.get(model_name, 0.5)
        
        print(f"\n{model_name}:")
        print(f"  new model F1: {new_f1:.4f}")
        
        # if no existing model, save directly
        if not has_existing_models or model_name not in existing_models:
            print(f"  ✅ no existing {model_name} model, save new model")
            models_to_save[model_name] = new_model
            thresholds_to_save[model_name] = new_threshold
            f1_scores_to_save[model_name] = new_f1
        else:
            # get existing model F1 score
            existing_f1 = existing_f1_scores.get(model_name, 0.0)
            
            print(f"  existing model F1: {existing_f1:.4f}")
            
            # compare F1 scores
            if new_f1 > existing_f1:
                improvement = new_f1 - existing_f1
                print(f"  ✅ new model better (improvement {improvement:.4f}), save new model")
                models_to_save[model_name] = new_model
                thresholds_to_save[model_name] = new_threshold
                f1_scores_to_save[model_name] = new_f1
            else:
                decline = existing_f1 - new_f1
                print(f"  ❌ new model worse (decline {decline:.4f}), keep existing model")
                # keep existing model F1 score
                f1_scores_to_save[model_name] = existing_f1
    
    # save models to update
    if models_to_save:
        print(f"\n💾 save {len(models_to_save)} improved models...")
        
        # if there are existing models, need to merge
        if has_existing_models:
            # keep existing models that are not updated
            final_models = existing_models.copy()
            final_thresholds = existing_thresholds.copy()
            final_f1_scores = existing_f1_scores.copy()
            
            # update improved models
            final_models.update(models_to_save)
            final_thresholds.update(thresholds_to_save)
            final_f1_scores.update(f1_scores_to_save)
            
            # use new scaler (because features may have changed)
            save_trained_models(final_models, new_scaler, final_thresholds, save_path)
        else:
            # no existing model, save new model directly
            final_f1_scores = f1_scores_to_save.copy()
            save_trained_models(models_to_save, new_scaler, thresholds_to_save, save_path)
        
        # save
        f1_scores_file = f"{save_path}model_f1_scores.pkl"
        with open(f1_scores_file, 'wb') as f:
            pickle.dump(final_f1_scores, f)
        print(f"✅ F1 scores saved: {f1_scores_file}")
        
    else:
        print("\n📊 no model needs update, all existing models are better")
        # even if no model needs update, save F1 scores (including existing)
        if f1_scores_to_save:
            f1_scores_file = f"{save_path}model_f1_scores.pkl"
            with open(f1_scores_file, 'wb') as f:
                pickle.dump(f1_scores_to_save, f)
            print(f"✅ F1 scores updated: {f1_scores_file}")

def load_trained_models(load_path="trained_models/"):
    """
    load trained models, scaler and optimal thresholds
    
    Returns:
    - models_dict: models dictionary
    - scaler: scaler
    - optimal_thresholds: optimal thresholds dictionary
    """
    import os
    
    print("=== load trained models ===")
    
    models_dict = {}
    model_files = {
        'Random Forest': 'random_forest_model.pkl',
        'XGBoost': 'xgboost_model.pkl',
        'SVC': 'svc_model.pkl',
        'Logistic Regression': 'logistic_regression_model.pkl'
    }
    
    # load models
    for model_name, filename in model_files.items():
        model_path = os.path.join(load_path, filename)
        if os.path.exists(model_path):
            with open(model_path, 'rb') as f:
                models_dict[model_name] = pickle.load(f)
            print(f"✅ {model_name} model loaded")
        else:
            print(f"⚠️ model file not found: {model_path}")
    
    # load scaler
    scaler_path = os.path.join(load_path, 'scaler.pkl')
    scaler = None
    if os.path.exists(scaler_path):
        with open(scaler_path, 'rb') as f:
            scaler = pickle.load(f)
        print(f"✅ scaler loaded")
    else:
        print(f"⚠️ scaler file not found: {scaler_path}")
    
    # load optimal thresholds
    thresholds_path = os.path.join(load_path, 'optimal_thresholds.pkl')
    optimal_thresholds = None
    if os.path.exists(thresholds_path):
        with open(thresholds_path, 'rb') as f:
            optimal_thresholds = pickle.load(f)
        print(f"✅ optimal thresholds loaded")
    else:
        print(f"⚠️ optimal thresholds file not found: {thresholds_path}")
    
    return models_dict, scaler, optimal_thresholds

def process_new_lmp_data_complete(da_file, rt_file, weather_file=None, output_path="processed_new_data/"):
    """
    process new LMP data (complete version: directly process time series data, similar to all_data in training)
    
    Parameters:
    - da_file: DA LMP data file path
    - rt_file: RT LMP data file path
    - weather_file: weather data file path (optional)
    - output_path: output path
    """
    import os
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    
    print("=== process new LMP data (complete version) ===")
    
    # read LMP data
    try:
        da_data = pd.read_csv(da_file)
        rt_data = pd.read_csv(rt_file)
        print(f"DA data: {len(da_data)} rows")
        print(f"RT data: {len(rt_data)} rows")
    except Exception as e:
        print(f"❌ read LMP data failed: {e}")
        return None
    
    # merge DA and RT data
    if len(da_data) != len(rt_data):
        print("⚠️ DA and RT data rows do not match, take the smaller length")
        min_len = min(len(da_data), len(rt_data))
        da_data = da_data[:min_len].reset_index(drop=True)
        rt_data = rt_data[:min_len].reset_index(drop=True)
    
    # merge DA and RT data
    combined_lmp = da_data.copy()
    if 'total_lmp_rt' not in combined_lmp.columns:
        rt_col = 'total_lmp_rt' if 'total_lmp_rt' in rt_data.columns else rt_data.columns[-1]
        combined_lmp['total_lmp_rt'] = rt_data[rt_col]
    
    print(f"merged LMP data: {len(combined_lmp)} rows")
    print(f"data columns: {combined_lmp.columns.tolist()}")
    
    # extract hour information from time column and create one-hot encoding (similar to training)
    print("extract hour information and create one-hot encoding...")
    
    # parse time and extract hour
    try:
        # try to parse time format
        combined_lmp['datetime_parsed'] = pd.to_datetime(combined_lmp['datetime_beginning_utc'])
        combined_lmp['hour_24'] = combined_lmp['datetime_parsed'].dt.hour
        
        # convert to 12-hour format (similar to training)
        def convert_to_12h_format(hour_24):
            if hour_24 == 0:
                return "12AM"
            elif hour_24 < 12:
                return f"{hour_24}AM"
            elif hour_24 == 12:
                return "12PM"
            else:
                return f"{hour_24-12}PM"
        
        combined_lmp['hourTime'] = combined_lmp['hour_24'].apply(convert_to_12h_format)
        
        # show hour distribution
        hour_counts = combined_lmp['hourTime'].value_counts()
        print(f"hour distribution: {hour_counts}")
        
    except Exception as e:
        print(f"⚠️ time parsing failed: {e}")
        # if parsing failed, use default value
        combined_lmp['hourTime'] = "4AM"  # default hour
    
    # add hour one-hot encoding (similar to training)
    print("create hour one-hot encoding...")
    all_hours = ['12AM', '1AM', '2AM', '3AM', '4AM', '5AM', '6AM', '7AM', '8AM', '9AM', '10AM', '11AM',
                '12PM', '1PM', '2PM', '3PM', '4PM', '5PM', '6PM', '7PM', '8PM', '9PM', '10PM', '11PM']
    
    # use pandas get_dummies to create one-hot encoding (more efficient)
    hour_dummies = pd.get_dummies(combined_lmp['hourTime'], prefix='hour')
    
    # ensure all 24 hours have corresponding columns (even if data does not have)
    for h in all_hours:
        col_name = f'hour_{h}'
        if col_name not in hour_dummies.columns:
            hour_dummies[col_name] = 0
    
    # sort hour columns in correct order
    hour_cols = [f'hour_{h}' for h in all_hours]
    combined_lmp = pd.concat([combined_lmp, hour_dummies[hour_cols]], axis=1)
    
    # add weather features (using actual weather data if provided)
    print("add weather features...")
    weather_cols = [
        'apparent_temperature (°C)', 
        'wind_gusts_10m (km/h)', 
        'pressure_msl (hPa)',
        'soil_temperature_0_to_7cm (°C)', 
        'soil_moisture_0_to_7cm (m³/m³)'
    ]
    
    if weather_file and os.path.exists(weather_file):
        try:
            # read weather data
            if weather_file.endswith('.pkl'):
                weather_data = pd.read_pickle(weather_file)
            else:
                weather_data = pd.read_csv(weather_file)
            
            print(f"load weather data: {len(weather_data)} rows")
            
            # parse weather data time
            weather_data['datetime_parsed'] = pd.to_datetime(weather_data['time'])
            weather_data['datetime_rounded'] = weather_data['datetime_parsed'].dt.round('H')
            
            # parse LMP data time and round to hour
            combined_lmp['datetime_rounded'] = pd.to_datetime(combined_lmp['datetime_beginning_utc']).dt.round('H')
            
            # create weather data lookup dictionary
            weather_lookup = {}
            for _, row in weather_data.iterrows():
                time_key = row['datetime_rounded']
                weather_lookup[time_key] = {col: row[col] for col in weather_cols if col in row}
            
            # match weather data for each LMP row
            for col in weather_cols:
                combined_lmp[col] = 0  # initialize
            
            matched_count = 0
            for idx, row in combined_lmp.iterrows():
                time_key = row['datetime_rounded']
                if time_key in weather_lookup:
                    for col in weather_cols:
                        if col in weather_lookup[time_key]:
                            combined_lmp.loc[idx, col] = weather_lookup[time_key][col]
                    matched_count += 1
                else:
                    # if no exact match, use nearest weather data
                    if weather_lookup:
                        closest_time = min(weather_lookup.keys(), key=lambda x: abs((x - time_key).total_seconds()))
                        for col in weather_cols:
                            if col in weather_lookup[closest_time]:
                                combined_lmp.loc[idx, col] = weather_lookup[closest_time][col]
            
            print(f"✅ weather features added: {matched_count}/{len(combined_lmp)} records exact match")
            
        except Exception as e:
            print(f"⚠️ read weather data failed: {e}")
            print("use default weather values...")
            # use default values as fallback
            default_weather_values = {
                'apparent_temperature (°C)': 15.0,
                'wind_gusts_10m (km/h)': 20.0,
                'pressure_msl (hPa)': 1013.25,
                'soil_temperature_0_to_7cm (°C)': 12.0,
                'soil_moisture_0_to_7cm (m³/m³)': 0.3
            }
            for col, val in default_weather_values.items():
                combined_lmp[col] = val
    else:
        print("⚠️ no weather file provided, use default values")
        # use reasonable default values (based on historical average)
        default_weather_values = {
            'apparent_temperature (°C)': 15.0,  # 15 degrees
            'wind_gusts_10m (km/h)': 20.0,      # 20 km/h
            'pressure_msl (hPa)': 1013.25,      # standard atmospheric pressure
            'soil_temperature_0_to_7cm (°C)': 12.0,  # 12 degrees
            'soil_moisture_0_to_7cm (m³/m³)': 0.3    # 30%
        }
        for col, val in default_weather_values.items():
            combined_lmp[col] = val
        print("✅ weather features added (using default values)")
    
    # add other features
    print("add other features...")
    
    # holiday features
    combined_lmp = applyHoliday(combined_lmp, holidays)
    
    # calculate LMP deviation
    total_lmp_delta(combined_lmp)
    
    # add lag features (similar to training)
    combined_lmp['lagged_lmp_da'] = combined_lmp['total_lmp_da'].shift(1).fillna(combined_lmp['total_lmp_da'].iloc[0])
    # remove information leakage features: lagged_lmp_rt and lagged_delta
    # because RT data is not available at prediction time
    
    # add multiple lag features (if used in training)
    k = 7  # similar to training
    for i in range(1, k+1):
        combined_lmp[f'total_lmp_da_lag_{i}'] = combined_lmp['total_lmp_da'].shift(i).fillna(combined_lmp['total_lmp_da'].iloc[0])
    
    # use corrected anomaly detection method (based on relative difference instead of 95% quantile)
    print("use relative difference method to detect anomalies...")
    try:
        # use corrected addTarget function
        addTarget_corrected(combined_lmp, k=2.5)
        print(f"anomaly distribution based on relative difference: {combined_lmp['target_c'].value_counts().to_dict()}")
            
    except Exception as e:
        print(f"⚠️ relative difference anomaly detection failed: {e}")
        # fallback to historical baseline method
        try:
            baseline_stats = build_historical_baseline(k=2.5)
            
            if baseline_stats is not None:
                combined_lmp = detect_anomalies_with_baseline(combined_lmp, baseline_stats, k=2.5)
                combined_lmp['target_c'] = combined_lmp['is_anomaly'].astype(int)
                combined_lmp['target_r'] = combined_lmp['total_lmp_delta']
                print(f"anomaly distribution based on historical baseline: {combined_lmp['target_c'].value_counts().to_dict()}")
            else:
                # final fallback
                combined_lmp['target_c'] = 0
                combined_lmp['target_r'] = combined_lmp['total_lmp_delta']
                print("use default anomaly marking (all non-anomalies)")
        except Exception as e2:
            print(f"⚠️ historical baseline anomaly detection also failed: {e2}")
            combined_lmp['target_c'] = 0
            combined_lmp['target_r'] = combined_lmp['total_lmp_delta']
            print("use default anomaly marking (all non-anomalies)")
    
    # save processed complete data
    output_file = os.path.join(output_path, "processed_new_data_complete.pkl")
    combined_lmp.to_pickle(output_file)
    print(f"💾 complete data saved: {output_file} ({len(combined_lmp)} rows)")
    
    print(f"\n✅ new data processed: {len(combined_lmp)} rows")
    print(f"contains features: {len(combined_lmp.columns)} columns")
    
    # return dictionary format for compatibility
    return {"complete": combined_lmp}

def validate_model_with_new_data(processed_data_path="processed_new_data/", 
                                 models_path="trained_models/"):
    """
    validate new data with trained models
    
    Parameters:
    - processed_data_path: processed new data path
    - models_path: trained models path
    """
    import os
    
    print("=== validate new data with trained models ===")
    
    # load trained models
    models_dict, scaler, optimal_thresholds = load_trained_models(models_path)
    
    if not models_dict or scaler is None:
        print("❌ cannot load trained models")
        return None
    
    # load processed new data (simplified version: directly read complete data file)
    complete_data_file = os.path.join(processed_data_path, "processed_new_data_complete.pkl")
    
    if not os.path.exists(complete_data_file):
        print(f"❌ cannot find complete data file: {complete_data_file}")
        return None
    
    all_results = {}
    all_predictions = []
    
    try:
        # directly read complete processed data
        new_data = pd.read_pickle(complete_data_file)
        print(f"\n📊 validate complete data set: {len(new_data)} rows")
        
        if len(new_data) == 0:
            print(f"⚠️ data is empty")
            return None
        
        # ensure target column exists
        if 'target_c' not in new_data.columns:
            print(f"add target variable...")
            if 'total_lmp_delta' not in new_data.columns:
                total_lmp_delta(new_data)
            
            # for small data, simply set target variable
            if len(new_data) == 1:
                new_data['target_c'] = 0  # single data default to non-anomaly
                new_data['target_r'] = new_data['total_lmp_delta']
            else:
                try:
                    addTarget(new_data)
                except:
                    # if addTarget fails, manually add
                    new_data['target_c'] = 0
                    new_data['target_r'] = new_data['total_lmp_delta']
        
        # prepare features (using the same feature set and order as training)
        # get exact feature order from saved scaler
        try:
            import pickle
            with open(os.path.join(models_path, 'scaler.pkl'), 'rb') as f:
                saved_scaler = pickle.load(f)
            if hasattr(saved_scaler, 'feature_names_in_'):
                inputs = saved_scaler.feature_names_in_.tolist()
                print(f"use training feature order: {len(inputs)} features")
            else:
                raise ValueError("scaler does not save feature names")
        except Exception as e:
            print(f"⚠️ cannot get feature order from scaler: {e}")
            # alternative: manually specify exact order as in training
            inputs = [
                'lagged_lmp_da',  # 只保留DA的滞后特征，移除lagged_delta避免信息泄漏
                'apparent_temperature (°C)', 'wind_gusts_10m (km/h)', 'pressure_msl (hPa)',
                'soil_temperature_0_to_7cm (°C)', 'soil_moisture_0_to_7cm (m³/m³)',
                'isHoliday',
                'total_lmp_da_lag_1', 'total_lmp_da_lag_2', 'total_lmp_da_lag_3', 
                'total_lmp_da_lag_4', 'total_lmp_da_lag_5', 'total_lmp_da_lag_6', 'total_lmp_da_lag_7',
                # hour features in alphabetical order as in training
                'hour_10AM', 'hour_10PM', 'hour_11AM', 'hour_11PM', 'hour_12AM', 'hour_12PM',
                'hour_1AM', 'hour_1PM', 'hour_2AM', 'hour_2PM', 'hour_3AM', 'hour_3PM',
                'hour_4AM', 'hour_4PM', 'hour_5AM', 'hour_5PM', 'hour_6AM', 'hour_6PM',
                'hour_7AM', 'hour_7PM', 'hour_8AM', 'hour_8PM', 'hour_9AM', 'hour_9PM'
            ]
        
        # check if features exist, fill missing features with 0
        available_inputs = []
        for col in inputs:
            if col in new_data.columns:
                available_inputs.append(col)
            else:
                # for missing features, add default value (mainly hour encoding)
                if col.startswith('hour_'):
                    new_data[col] = 0
                    available_inputs.append(col)
                else:
                    print(f"⚠️ missing feature: {col}")
        
        print(f"available features: {len(available_inputs)}/{len(inputs)}")
        
        if len(available_inputs) < len(inputs) * 0.7:  # at least 70% features available
            print(f"⚠️ not enough available features, skip prediction")
            return None
        
        # use the same feature order as in training
        X_new = new_data[inputs]  # now all features should exist
        
        # check if there are any NaN values
        if X_new.isnull().any().any():
            print(f"⚠️ detected NaN values, fill with 0")
            X_new = X_new.fillna(0)
        
        # standardize features
        X_new_scaled = scaler.transform(X_new)
        y_true = new_data['target_c']
        
        complete_results = {}
        
        # use each model to predict
        for model_name, model in models_dict.items():
            try:
                # get prediction probabilities
                y_pred_proba = model.predict_proba(X_new_scaled)[:, 1]
                
                # use optimal threshold
                threshold = optimal_thresholds.get(model_name, 0.5)
                y_pred = (y_pred_proba >= threshold).astype(int)
                
                # calculate evaluation metrics
                accuracy = accuracy_score(y_true, y_pred)
                precision = precision_score(y_true, y_pred, zero_division=0)
                recall = recall_score(y_true, y_pred, zero_division=0)
                f1 = f1_score(y_true, y_pred, zero_division=0)
                auc = roc_auc_score(y_true, y_pred_proba) if len(np.unique(y_true)) > 1 else 0
                
                complete_results[model_name] = {
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'predictions': y_pred,
                    'probabilities': y_pred_proba,
                    'threshold': threshold
                }
                
                print(f"{model_name}: accuracy={accuracy:.3f}, F1={f1:.3f}, AUC={auc:.3f}")
                print(f"   predicted anomalies: {y_pred.sum()}/{len(y_pred)} ({y_pred.mean()*100:.1f}%)")
                print(f"   true anomalies: {y_true.sum()}/{len(y_true)} ({y_true.mean()*100:.1f}%)")
                
                # save prediction results
                prediction_record = {
                    'dataset': 'complete',
                    'model': model_name,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall,
                    'f1': f1,
                    'auc': auc,
                    'threshold': threshold,
                    'num_samples': len(y_true),
                    'num_anomalies_true': y_true.sum(),
                    'num_anomalies_pred': y_pred.sum()
                }
                all_predictions.append(prediction_record)
                
            except Exception as e:
                print(f"❌ {model_name} prediction failed: {e}")
        
        all_results['complete'] = complete_results
        
    except Exception as e:
        print(f"❌ process complete data failed: {e}")
        import traceback
        traceback.print_exc()
        return None
    
    # save validation results
    if all_predictions:
        results_df = pd.DataFrame(all_predictions)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = f"model_validation_results_{timestamp}.csv"
        results_df.to_csv(results_file, index=False)
        print(f"\n💾 validation results saved: {results_file}")
        
        # show overall results summary
        print(f"\n📈 validation results summary:")
        summary = results_df.groupby('model').agg({
            'accuracy': 'mean',
            'precision': 'mean', 
            'recall': 'mean',
            'f1': 'mean',
            'auc': 'mean'
        }).round(3)
        print(summary)
    
    return all_results
#%%
# check if files exist
def check_files_exist(da_file, rt_file, weather_path):
    """check if necessary files exist"""
    import os
    
    print("=== check file status ===")
    
    # check DA file
    if os.path.exists(da_file):
        print(f"✅ DA file exists: {da_file}")
    else:
        print(f"❌ DA file does not exist: {da_file}")
        return False
    
    # check RT file  
    if os.path.exists(rt_file):
        print(f"✅ RT file exists: {rt_file}")
    else:
        print(f"❌ RT file does not exist: {rt_file}")
        return False
    
    # check weather data directory
    if os.path.exists(weather_path):
        weather_files = [f for f in os.listdir(weather_path) if f.startswith('western_hub_weather')]
        print(f"✅ weather data directory exists: {weather_path} ({len(weather_files)} files)")
        if weather_files:
            print(f"   example files: {weather_files[:3]}")
        return len(weather_files) > 0
    else:
        print(f"❌ weather data directory does not exist: {weather_path}")
        return False

# %%

# %%
# run complete model application process

print("=== complete model application process ===")

# first step: check if necessary files exist
print("first step: check necessary files")
files_ok = check_files_exist(
    da_file="applicationData/da_hrl_lmps_2025.csv",
    rt_file="applicationData/rt_hrl_lmps_2025.csv", 
    weather_path="weatherData/meteo/"
)

if not files_ok:
    print("❌ necessary files do not exist, please check file paths")
else:
    print("✅ all necessary files exist")
    
    # second step: if no trained models, train and save models
    import os
    if not os.path.exists("trained_models/"):
        print("\nsecond step: train and save models (first run)")
        
        # retrain models (using previous code)
        print("retrain models...")
        
        # use previous all_data and features
        optimal_thresholds = {
            'Random Forest': optimal_threshold_rf,
            'XGBoost': optimal_threshold_xgb, 
            'SVC': optimal_threshold_svc,
            'Logistic Regression': optimal_threshold_lr
        }
        
        models_dict = {
            'Random Forest': rf,
            'XGBoost': xgb,
            'SVC': svc,
            'Logistic Regression': best_lr
        }
        
        # save trained models
        save_trained_models(models_dict, scaler, optimal_thresholds)
        
    else:
        print("\nsecond step: found existing trained models")
    
    # third step: process new data
    print("\nthird step: process new LMP data")
    
    # use complete processing function (using actual weather data)
    processed_data = process_new_lmp_data_complete(
        da_file="applicationData/da_hrl_lmps_2025.csv",
        rt_file="applicationData/rt_hrl_lmps_2025.csv",
        weather_file="weatherData/meteo/western_hub_weather_2025-01-01_to_2025-06-06.pkl"
    )
    
    if processed_data:
        # fourth step: use trained models to validate new data  
        print("\nfourth step: use trained models to predict new data")
        validation_results = validate_model_with_new_data()
        
        if validation_results:
            print("\n🎯 model application completed!")
            print("- used trained models to predict new data")
            print("- validation results saved to CSV file")
            print("- can view performance of each model on new data")
        else:
            print("❌ model validation failed")
    else:
        print("❌ new data processing failed")



# %%
import pandas as pd

#file_path = 'raw_data/rt_hrl_lmps_2022.csv'
#file_path = 'raw_data/rt_hrl_lmps_2023_1-6.csv'
#file_path = 'raw_data/da_hrl_lmps_2023_1-6.csv'
file_path = 'raw_data/da_hrl_lmps_2022.csv'

print(f"reading data from {file_path}...")
try:
    # The head output shows some formatting issues. Let's try to be robust.
    df = pd.read_csv(file_path, skipinitialspace=True)
except Exception as e:
    print(f"error reading CSV: {e}")
    # Fallback if the first line is problematic
    try:
        df = pd.read_csv(file_path, skipinitialspace=True, on_bad_lines='skip')
    except Exception as e2:
        print(f"error reading CSV again: {e2}")
        exit()


print(f"original rows: {len(df)}")

# The user wants to filter for pnode=51288
# The column name is 'pnode_id'.
pnode_to_keep = 51288
df_filtered = df[df['pnode_id'] == pnode_to_keep]

print(f"rows after filtering pnode_id {pnode_to_keep}: {len(df_filtered)}")

# Overwriting the original file with the filtered data as requested.
print(f"saving filtered data back to {file_path}...")
df_filtered.to_csv(file_path, index=False)

print("filtering completed.")

# %%
import pandas as pd
import os

# define file paths
data_dir = 'raw_data'
file1_path = os.path.join(data_dir, 'da_hrl_lmps_2023_1-6.csv')
file2_path = os.path.join(data_dir, 'da_hrl_lmps_2023_7-12.csv')
output_file_path = os.path.join(data_dir, 'da_hrl_lmps_2023.csv')

print(f"reading data from {file1_path}...")
df1 = pd.read_csv(file1_path)
print(f"read {len(df1)} rows.")

print(f"reading data from {file2_path}...")
df2 = pd.read_csv(file2_path)
print(f"read {len(df2)} rows.")

print("merging two data frames...")
merged_df = pd.concat([df1, df2], ignore_index=True)
print(f"merged total rows: {len(merged_df)}")

print(f"saving merged data to {output_file_path}...")
merged_df.to_csv(output_file_path, index=False)

print("file merging completed.")

# %%
