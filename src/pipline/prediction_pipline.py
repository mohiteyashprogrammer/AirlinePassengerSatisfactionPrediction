import os
import sys
import pandas as pd 
import numpy as np 
from src.logger import logging
from src.exception import CustomException
from dataclasses import dataclass

from src.utils import load_object


class PredictPipline:
    def __init__(self):
        pass


    def predict(self,features):
        try:
            # Load Pickel File 
            ## This Code Work in /any system
            preprocessor_path = os.path.join("artifcats","preprocessor.pkl")
            model_path = os.path.join("artifcats","model.pkl")

            ## Load Pickel File 
            preprocessor = load_object(preprocessor_path)
            model = load_object(model_path)

            data_scaled = preprocessor.transform(features)

            pred = model.predict(data_scaled)

            return pred

        except Exception as e:
            logging.info("Error Occure in Prediction Pipline")
            raise CustomException(e, sys)


class CustomData:
    def __init__(self,
            Gender:int,
            Age:int,
            Type_of_Travel:str,
            Class:str,
            Flight_Distance:int,
            Inflight_wifi_service:int,
            Ease_of_Online_booking:int,
            Food_and_drink:int,
            Online_boarding:int,
            Seat_comfort:int,
            Inflight_entertainment:int,
            On_board_service:int,
            Leg_rooms_service:int,
            Baggage_handling:int,
            Checkin_service:int,
            Inflight_service:int,
            Cleanliness:int):
            

        self.Gender = Gender
        self.Age = Age
        self.Type_of_Travel = Type_of_Travel
        self.Class = Class
        self.Flight_Distance = Flight_Distance
        self.Inflight_wifi_service = Inflight_wifi_service
        self.Ease_of_Online_booking = Ease_of_Online_booking
        self.Food_and_drink = Food_and_drink
        self.Online_boarding = Online_boarding
        self.Seat_comfort = Seat_comfort
        self.Inflight_entertainment = Inflight_entertainment
        self.On_board_service = On_board_service
        self.Leg_rooms_service = Leg_rooms_service
        self.Baggage_handling = Baggage_handling
        self.Checkin_service = Checkin_service
        self.Inflight_service = Inflight_service
        self.Cleanliness = Cleanliness


    def get_data_as_data_frame(self):
        try:
            custom_data_input_dict = {
                "Gender":[self.Gender],
                "Age":[self.Age],
                "Type_of_Travel":[self.Type_of_Travel],
                "Class":[self.Class],
                "Flight_Distance":[self.Flight_Distance],
                "Inflight_wifi_service":[self.Inflight_wifi_service],
                "Ease_of_Online_booking":[self.Ease_of_Online_booking],
                "Food_and_drink":[self.Food_and_drink],
                "Online_boarding":[self.Online_boarding],
                "Seat_comfort":[self.Seat_comfort],
                "Inflight_entertainment":[self.Inflight_entertainment],
                "On_board_service":[self.On_board_service],
                "Leg_rooms_service":[self.Leg_rooms_service],
                "Baggage_handling":[self.Baggage_handling],
                "Checkin_service":[self.Checkin_service],
                "Inflight_service":[self.Inflight_service],
                "Cleanliness":[self.Cleanliness]
            }

            data = pd.DataFrame(custom_data_input_dict)
            logging.info("DataFrame Gathered")
            return data

        except Exception as e:
            logging.info("Error Occured In Predict Pipline")
            raise CustomException(e, sys)