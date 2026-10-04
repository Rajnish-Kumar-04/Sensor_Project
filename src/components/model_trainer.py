import sys 
from typing import Generator,Tuple,List
import os
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier,GradientBoostingClassifier,AdaBoostClassifier
from sklearn.model_selection import train_test_split

from src.constant import * 
from src.exception import CustomException
from src.logger import logging
from src.utils.main_utils import MainUtils
from dataclasses import dataclass

@dataclass
class ModelTrainerConfig:
    artifact_folder = os.path.join(artifact_folder)
    trained_model_path = os.path.join(artifact_folder, "model.pkl")
    expected_accuracy = 0.45
    model_config_file_path = os.path.join('config', "model.yaml")

class ModelTrainer:

    def __init__(self):

        self.model_trainer_config = ModelTrainerConfig()

        self.utils = MainUtils()


        self.models = {
            "XGBClassifier": XGBClassifier(),
            "RandomForestClassifier": RandomForestClassifier(),
            "GradientBoostingClassifier": GradientBoostingClassifier(),
            "AdaBoostClassifier": AdaBoostClassifier()
        }

    def evaluate_models(self, X,y,models):
        try:
            X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,random_state=42)
            report = {}

            for i in range(len(list(models))):
                model = list(models.values())[i]
                model.fit(X_train, y_train)

                y_train_pred = model.predict(X_train)

                y_test_pred = model.predict(X_test)

                train_model_score = accuracy_score(y_train, y_train_pred)

                test_model_score = accuracy_score(y_test, y_test_pred)

                report[list(models.keys())[i]] = test_model_score

            return report
        except Exception as e:  
            raise CustomException(e,sys)

    def get_best_model(self, 
                       x_train = np.array,
                       y_train = np.array,
                       x_test = np.array,
                       y_test = np.array):
        try:

            model_report:dict = self.evaluate_models(x_train = x_train, y_train = y_train,
                                                     x_test = x_test, y_test = y_test, 
                                                     models = self.models)
            

            print(f"Model Report : {model_report}")

            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]

            best_model_object = self.models[best_model_name]

            return best_model_name, self.models.object, best_model_score
        except Exception as e:
            raise CustomException(e,sys)

    def finetune_best_model(self,best_model_object,
                            best_model_name,
                            X_train,
                            y_train) -> object:
        try:

            model_param_grid = self.utils.read_yaml_file(self.model_trainer_config.model_config_file_path)[""]
                
        