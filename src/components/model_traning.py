import os
import sys
import pandas as pd 
import numpy as np 
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.utils import save_object
from src.utils import model_evalution

@dataclass
class ModelTraningConfig:
    train_model_file_obj = os.path.join("artifcats","model.pkl")


class ModelTraning:
    def __init__(self):
        self.model_traner_config = ModelTraningConfig()


    def initatied_model_traning(self,train_array,test_array):
        try:
            logging.info("Split Dependent And Independent Features")
            X_train,y_train,X_test,y_test = (
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )


            ## Model Traning
            models = {
            "LogisticRegression":LogisticRegression(),
            "RandomForestClassifier":RandomForestClassifier()
            }

            model_report:dict = model_evalution(X_train, y_train, X_test, y_test, models)
            print(model_report)
            print("\n*************************************************************************************\n")
            logging.info(f"Model Report: {model_report}")

            ## To Get The Best Model Score From Dict
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            

            best_model = models[best_model_name]

            print(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")
            print("\n***************************************************************************************\n")
            logging.info(f"Best Model Found, Model Name is: {best_model_name},Accuracy_Score: {best_model_score}")


            save_object(file_path=self.model_traner_config.train_model_file_obj,
            obj = best_model
            )

        except Exception as e:
            logging.info("Error Occured in Model Traning")
            raise CustomException(e,sys)


