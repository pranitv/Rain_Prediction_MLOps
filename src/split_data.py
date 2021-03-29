# split the raw data and save in data\processed
import os
import argparse
import pandas as pd
from sklearn.model_selection import train_test_split
from get_data import read_params

def randomsampleimputation(df, variable):
    df[variable]=df[variable]
    random_sample=df[variable].dropna().sample(df[variable].isnull().sum(),random_state=0)
    random_sample.index=df[df[variable].isnull()].index
    df.loc[df[variable].isnull(),variable]=random_sample

def Handle_missing_values(df):
    numerical_feature = [feature for feature in df.columns if df[feature].dtypes != 'O']
    discrete_feature=[feature for feature in numerical_feature if len(df[feature].unique())<25]
    continuous_feature = [feature for feature in numerical_feature if feature not in discrete_feature]
    categorical_feature = [feature for feature in df.columns if feature not in numerical_feature]
    for i in ['Cloud9am','Cloud3pm','Evaporation','Sunshine']:
        randomsampleimputation(df, i)
    for feature in continuous_feature:
        if(df[feature].isnull().sum()*100/len(df))>0:
            df[feature] = df[feature].fillna(df[feature].median())
    for feature in discrete_feature:
        mode=df[feature].value_counts().index[0]
        df[feature].fillna(mode,inplace=True)
    
    df["RainToday"] = pd.get_dummies(df["RainToday"], drop_first = True)
    df["RainTomorrow"] = pd.get_dummies(df["RainTomorrow"], drop_first = True)

    windgustdir = {'NNW':0, 'NW':1, 'WNW':2, 'N':3, 'W':4, 'WSW':5, 'NNE':6, 'S':7, 'SSW':8, 'SW':9, 'SSE':10,
       'NE':11, 'SE':12, 'ESE':13, 'ENE':14, 'E':15}
    winddir9am = {'NNW':0, 'N':1, 'NW':2, 'NNE':3, 'WNW':4, 'W':5, 'WSW':6, 'SW':7, 'SSW':8, 'NE':9, 'S':10,
       'SSE':11, 'ENE':12, 'SE':13, 'ESE':14, 'E':15}
    winddir3pm = {'NW':0, 'NNW':1, 'N':2, 'WNW':3, 'W':4, 'NNE':5, 'WSW':6, 'SSW':7, 'S':8, 'SW':9, 'SE':10,
       'NE':11, 'SSE':12, 'ENE':13, 'E':14, 'ESE':15}
    df["WindGustDir"] = df["WindGustDir"].map(windgustdir)
    df["WindDir9am"] = df["WindDir9am"].map(winddir9am)
    df["WindDir3pm"] = df["WindDir3pm"].map(winddir3pm)

    df["WindGustDir"] = df["WindGustDir"].fillna(df["WindGustDir"].value_counts().index[0])
    df["WindDir9am"] = df["WindDir9am"].fillna(df["WindDir9am"].value_counts().index[0])
    df["WindDir3pm"] = df["WindDir3pm"].fillna(df["WindDir3pm"].value_counts().index[0])

    location = {'Portland':1, 'Cairns':2, 'Walpole':3, 'Dartmoor':4, 'MountGambier':5,
       'NorfolkIsland':6, 'Albany':7, 'Witchcliffe':8, 'CoffsHarbour':9, 'Sydney':10,
       'Darwin':11, 'MountGinini':12, 'NorahHead':13, 'Ballarat':14, 'GoldCoast':15,
       'SydneyAirport':16, 'Hobart':17, 'Watsonia':18, 'Newcastle':19, 'Wollongong':20,
       'Brisbane':21, 'Williamtown':22, 'Launceston':23, 'Adelaide':24, 'MelbourneAirport':25,
       'Perth':26, 'Sale':27, 'Melbourne':28, 'Canberra':29, 'Albury':30, 'Penrith':31,
       'Nuriootpa':32, 'BadgerysCreek':33, 'Tuggeranong':34, 'PerthAirport':35, 'Bendigo':36,
       'Richmond':37, 'WaggaWagga':38, 'Townsville':39, 'PearceRAAF':40, 'SalmonGums':41,
       'Moree':42, 'Cobar':43, 'Mildura':44, 'Katherine':45, 'AliceSprings':46, 'Nhil':47,
       'Woomera':48, 'Uluru':49}
    df["Location"] = df["Location"].map(location)

    df["Date"] = pd.to_datetime(df["Date"], format = "%Y-%m-%dT", errors = "coerce")
    df["Date_month"] = df["Date"].dt.month
    df["Date_day"] = df["Date"].dt.day

    return df

def Handle_outliers(df):
    continuous_feature = ['MinTemp','MaxTemp','Rainfall','Evaporation','WindGustSpeed','WindSpeed9am','WindSpeed3pm','Humidity9am',
    'Pressure9am','Pressure3pm','Temp9am','Temp3pm']
    for feature in continuous_feature:
        IQR=df[feature].quantile(0.75)-df[feature].quantile(0.25)
        lower_bridge=df[feature].quantile(0.25)-(IQR*1.5)
        upper_bridge=df[feature].quantile(0.75)+(IQR*1.5)
        df.loc[df[feature]>=round(upper_bridge,2),feature]=round(upper_bridge,2)
        df.loc[df[feature]<=round(lower_bridge,2),feature]=round(lower_bridge,2)

    return df


def split_and_save_data(config_path):
    config = read_params(config_path)
    test_data_path = config['split_data']['test_path'] 
    train_data_path = config['split_data']['train_path']
    raw_data_path = config['load_data']['raw_dataset_csv']
    split_size = config['split_data']['test_size']
    random_state = config['base']['random_state']

    df = pd.read_csv(raw_data_path, sep =',')
    df = Handle_missing_values(df)
    df = Handle_outliers(df)
    train,test = train_test_split(df,test_size = split_size,random_state = random_state)

    train.to_csv(train_data_path,sep=',',index=False)
    test.to_csv(test_data_path,sep=',',index = False)


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    split_and_save_data(config_path=parsed_args.config)