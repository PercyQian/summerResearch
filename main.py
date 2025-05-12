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
def total_lmp_delta(data : pd.DataFrame) -> None: 
    data['total_lmp_delta']  = 0
    for  i in list(data.index):
        data.loc[i, "total_lmp_delta"] = data.loc[i, "total_lmp_da"] - data.loc[i, "total_lmp_rt"]

# %%
def addTarget(df : pd.DataFrame)->None:
    num = len(df['total_lmp_delta'])
    top_5_percent_index = num * 0.95
    bottom_95_percent_index = num * 0.05

    min_5 = sorted(df['total_lmp_delta'])[:int(bottom_95_percent_index)][-1]
    max_5 = sorted(df['total_lmp_delta'])[int(top_5_percent_index):][0]

    df['target_c'] = 0

    # Can also just classify outliers as 1, instead of separating into positive and negative
    for i in list(reversed(df.index)):

        if df['total_lmp_delta'][i] <= min_5:
            df.loc[i,'target_c'] = 2
        elif df['total_lmp_delta'][i] >= max_5:
            df.loc[i,'target_c'] = 1
        else: 
            df.loc[i,'target_c'] = 0

    df['target_c'] = df['target_c'].shift(-1)
    df['target_r'] = df['total_lmp_delta'].shift(-1)
    df.dropna(inplace=True)

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


# %%
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline

# %%
com = loadDataByHour("8PM")


inputs = (com.describe().columns[:-2])
outputs = (com.describe().columns[-2:])

# %%
def plot_features(model, model_params = None) ->list:
    com = loadDataByHour("8PM")
    inputs = (com.describe().columns[:-2])
    outputs = (com.describe().columns[-2:])

    scaler = StandardScaler()
    scaler.fit(com[inputs])
    X = scaler.transform(com[inputs])

    X_train, X_test, y_train, y_test = train_test_split(X, com[outputs]["target_c"], test_size=0.2, shuffle=False)

    model.fit(X_train, y_train)

    r = permutation_importance(model, X_train, y_train, n_repeats=1, random_state=42)

    features = []
    values = []

    for i in r.importances_mean.argsort()[::-1]:
        print(f"{inputs[i]:<30} importance: {r.importances_mean[i]:.5f} +/- {r.importances_std[i]:.5f}" )
        features.append(inputs[i])
        values.append(r.importances_mean[i])

    fig, ax = plt.subplots()
    fig.set_figheight(12)
    fig.set_figwidth(16)
    y_pos = np.arange(len(features))

    ax.barh(y_pos,values,align="center",color ='maroon')
    ax.set_yticks(y_pos,labels=features)
    ax.invert_yaxis() 

    ax.set_xlabel("Features")
    ax.set_ylabel("Feature Importance")
    ax.set_title("Importance of Features on target")
    plt.show()

    return dict(zip(features, values))

# %%
rlf_features = plot_features(RandomForestClassifier())

# %%
rlf_features = [i for i in list(rlf_features.keys()) if rlf_features[i] > 0]


scaler = StandardScaler()
scaler.fit(com[rlf_features])
X = scaler.transform(com[rlf_features])


X_train, X_test, y_train, y_test = train_test_split(X,com[outputs]["target_c"],test_size=0.2, shuffle=False)

# 打印原始类别分布
print("\n训练集原始类别分布:")
for i in range(3):
    print(f"类别 {i}: {np.sum(y_train == i)} 样本")

# 创建采样策略
# 使用字典指定目标样本数
over = SMOTE(sampling_strategy={
    1: 200,  # 增加正异常的样本数
    2: 200   # 增加负异常的样本数
})
under = RandomUnderSampler(sampling_strategy={
    0: 300   # 增加正常样本的数量
})

# RandomForest - 修改参数
rlf_params = {
    'criterion': 'entropy',  # 使用信息熵作为分裂标准
    'min_samples_leaf': 2,
    'n_estimators': 300,    # 增加树的数量
    'max_depth': 20,        # 增加树的深度
    'class_weight': 'balanced',
    'min_samples_split': 5  # 添加最小分裂样本数
}

# 创建包含采样和模型的管道
rlf_pipeline = ImbPipeline([
    ('over', over),
    ('under', under),
    ('classifier', RandomForestClassifier(**rlf_params))
])

# 训练模型
rlf_pipeline.fit(X_train, y_train)
y_pred = rlf_pipeline.predict(X_test)

print("\nRandom Forest 评估结果:")
evaluate_anomaly_detection(y_test, y_pred, "Random Forest")

# XGBoost - 修改参数
xgb_params = {
    'colsample_bytree': 0.8,
    'gamma': 0.1,           # 减小gamma值，使模型更容易分裂
    'max_depth': 10,        # 增加树的深度
    'min_child_weight': 1,
    'subsample': 0.8,
    'scale_pos_weight': 20, # 增加正样本权重
    'learning_rate': 0.05,  # 添加学习率
    'n_estimators': 300     # 增加树的数量
}

xgb_pipeline = ImbPipeline([
    ('over', over),
    ('under', under),
    ('classifier', XGBClassifier(**xgb_params))
])

xgb_pipeline.fit(X_train, y_train)
y_pred = xgb_pipeline.predict(X_test)

print("\nXGBoost 评估结果:")
evaluate_anomaly_detection(y_test, y_pred, "XGBoost")

# SVC - 修改参数
svc_params = {
    'C': 20.0,             # 增加C值，使决策边界更灵活
    'probability': True,
    'class_weight': 'balanced',
    'kernel': 'rbf',
    'gamma': 'scale',
    'cache_size': 1000     # 增加缓存大小
}

svc_pipeline = ImbPipeline([
    ('over', over),
    ('under', under),
    ('classifier', SVC(**svc_params))
])

svc_pipeline.fit(X_train, y_train)
y_pred = svc_pipeline.predict(X_test)

print("\nSVC 评估结果:")
evaluate_anomaly_detection(y_test, y_pred, "SVC")

# Logistic Regression - 修改参数
lr_params = {
    'class_weight': 'balanced',
    'max_iter': 3000,      # 增加最大迭代次数
    'C': 20.0,             # 增加正则化强度的倒数
    'solver': 'saga',
    'tol': 1e-4            # 调整收敛容差
}

lr_pipeline = ImbPipeline([
    ('over', over),
    ('under', under),
    ('classifier', LogisticRegression(**lr_params))
])

lr_pipeline.fit(X_train, y_train)
y_pred = lr_pipeline.predict(X_test)

print("\nLogistic Regression 评估结果:")
evaluate_anomaly_detection(y_test, y_pred, "Logistic Regression")

# 打印测试集分布
print("\n测试集类别分布:")
for i in range(3):
    print(f"类别 {i}: {np.sum(y_test == i)} 样本")

# 打印采样后的训练集分布
y_train_resampled = rlf_pipeline.named_steps['classifier'].predict(X_train)
print("\n采样后的训练集类别分布:")
for i in range(3):
    print(f"类别 {i}: {np.sum(y_train_resampled == i)} 样本")

# %%
