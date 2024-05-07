import os
import sys
from src.SensorFaultDetection.exception import CustomException
from src.SensorFaultDetection.logger import logging
import pandas as pd
import numpy as np
from pymongo import MongoClient
from src.SensorFaultDetection.utils import export_collection_as_dataframe
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from dotenv import load_dotenv

load_dotenv()

url=os.getenv("MONGO_DB_URL")
db=os.getenv("db_name")
coll=os.getenv("collection_name")


@dataclass
class DataIngestionConfig:
    train_data_path:str=os.path.join('artifacts','train.csv')
    test_data_path:str=os.path.join('artifacts','test.csv')
    raw_data_path:str=os.path.join('artifacts','raw.csv')        

class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initiate_data_ingestion(self):
        try:
            logging.info("initiated data ingestion")
            df=export_collection_as_dataframe(url,coll,db)
            logging.info("Reading data from database")
            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)
            train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
            train_set.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_set.to_csv(self.ingestion_config.test_data_path,index=False,header=True)
            logging.info("Data Ingestion is completed")

            return(
                self.ingestion_config.train_data_path,
                self.ingestion_config.test_data_path


            )

        except Exception as e:
            raise CustomException(e, sys)

