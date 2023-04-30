import os
import sys
import pandas as pd 
import numpy as np 
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder 
from sklearn.preprocessing import OneHotEncoder
# Pipline
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer

from src.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path = os.path.join("artifcats","preprocessor.pkl")


#Create Data TRansdormation Class
class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()


    def get_data_transformation_object(self):
        try:
    
            logging.info("Data Transformation Started")

            catigorical_features = ['Type_of_Travel', 'Class']

            numerical_features = ['Gender', 'Age', 'Flight_Distance', 'Inflight_wifi_service',
            'Ease_of_Online_booking', 'Food_and_drink', 'Online_boarding',
            'Seat_comfort', 'Inflight_entertainment', 'On_board_service',
            'Leg_rooms_service', 'Baggage_handling', 'Checkin_service',
            'Inflight_service', 'Cleanliness']

            logging.info("Pipline Started")

            #Numerical Pipline
            num_pipline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="median")),
                ("scaler",StandardScaler())
            ]
        )

            cato_pipline = Pipeline(
                steps=[
                ("imputer",SimpleImputer(strategy="most_frequent")),
                ("onehote",OneHotEncoder()),
                ("scaler",StandardScaler(with_mean=False))
            ]
        )

            ## Get Preprocessing Object
            preprocessing = ColumnTransformer([
            ("num_pipline",num_pipline,numerical_features),
            ("cato_pipline",cato_pipline,catigorical_features)
    
            ])

            return preprocessing

        except Exception as e:
            logging.info("Error Occured In Data TRansformation Class")
            raise CustomException(e, sys)

    

    def inited_data_transformation(self,train_path,test_path):
        try:
            ## Read Train And Test Data
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)

            logging.info("Read Traning and TEst Data Complited")
            logging.info(f"Train DataFrame Head : \n{train_data.head().to_string()}")
            logging.info(f"Test DataFrame Head : \n{test_data.head().to_string()}")

            logging.info("Obtaning Prrprocessor Object")

            preprocessor_obj = self.get_data_transformation_object()

            target_columns_name = "satisfaction"
            drop_column = [target_columns_name]

            ## Split dependent nd independent features
            input_features_train_data = train_data.drop(drop_column,axis=1)
            target_feature_train_data = train_data[target_columns_name]

            ## Split dependent nd independent features
            input_features_test_data = test_data.drop(drop_column,axis=1)
            target_feature_test_data = test_data[target_columns_name]
    

            ## Apply Transformton Object on train and test data
            input_features_train_arr = preprocessor_obj.fit_transform(input_features_train_data)
            input_feature_test_arr = preprocessor_obj.transform(input_features_test_data)


            logging.info("Apply Preprocessor Object on train and test Data")

            ## Convert in To Array To BEcome Faster
            train_array = np.c_[input_features_train_arr,np.array(target_feature_train_data)]
            test_array = np.c_[input_feature_test_arr,np.array(target_feature_test_data)]
            

            # CAlling Saving Object and Asve Preprocessing Object
            save_object(file_path = self.data_transformation_config.preprocessor_obj_file_path,
            obj = preprocessor_obj
            )

            logging.info("Preproccsing Pickel File Save")

            return (
                train_array,
                test_array,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception  as e:
            logging.info("Error Occured In Data Transformation Class")
            raise CustomException(e, sys)
