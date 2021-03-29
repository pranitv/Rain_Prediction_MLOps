# load train and test file
# train algo
# save metrics and prams

import os
import warnings
import sys
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
from imblearn.over_sampling import SMOTE
from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
from sklearn import metrics
from sklearn.ensemble import RandomForestClassifier
from catboost import CatBoostClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from get_data import read_params
from urllib.parse import urlparse
import argparse
import joblib
import json
import mlflow


# def evaluate_metrics(actual,pred):
#     rmse = np.sqrt(mean_squared_error(actual,pred))
#     mae = mean_absolute_error(actual,pred)
#     r2 = r2_score(actual,pred)
#     return rmse,mae,r2

def train_and_evaluate(config_path):
    config = read_params(config_path)
    train_data_path = config['split_data']['train_path']
    test_data_path = config['split_data']['test_path'] 
    random_state = config['base']['random_state']
    model_dir = config['model_dir']

    
    
    target = [config['base']['target_col']]

    train = pd.read_csv(train_data_path,sep=',')
    test = pd.read_csv(test_data_path,sep=',')

    train_y =train[target] 
    test_y = test[target]

    train_X = train.drop([target,'Date'], axis=1)
    test_X = test.drop([target,'Date'], axis=1)

    sm=SMOTE(random_state=0)
    X_train_res, y_train_res = sm.fit_resample(train_X, train_y)

    

    # #################### MLFLOW #################################
    # mlflow_config = config('mlflow config')
    # remote_server_url = mlflow_config['remote_server_url']

    # mlflow.set_tracking_url(remote_server_url)
    # mlflow.set_experiment(mlflow_config['experiment_name'])
    
    # with mlflow.start_run(run_name=mlflow_config['run_name']) as mlops_run:
    #     lr = ElasticNet(alpha = alpha, l1_ratio=l1_ratio, random_state=random_state)
    #     lr.fit(train_X,train_y)

    #     predicted_quality = lr.predict(test_X)
    #     (rmse, mae, r2) = evaluate_metrics(test_y,predicted_quality)
    #     mlflow.log_param('alpha',alpha)
    #     mlflow.log_param('l1_ratio',l1_ratio)
        
    #     mlflow.log_metric('rmse',rmse)
    #     mlflow.log_metric('mae',mae)
    #     mlflow.log_metric('r2_score',r2)

    #     tracking_url_type_store = urlparse(mlflow.get_artifact_url()).scheme

    #     if tracking_url_type_store != "file":
    #         mlflow.sklearn.log_model(lr,"model",registered_model_name =mlflow_config['registered_model_name'])
    #     else:
    #         mlflow.sklearn.load_model(lr,'model')



#####################################################
       #####################################################


    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, "model.joblib")

    joblib.dump(lr, model_path)


    


if __name__=="__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--config", default="params.yaml")
    parsed_args = args.parse_args()
    train_and_evaluate(config_path=parsed_args.config)