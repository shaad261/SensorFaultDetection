import os
import sys
import boto3
import dill
import numpy as np
import pandas as pd
from pymongo import MongoClient
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from src.SensorFaultDetection.exception import CustomException
from src.SensorFaultDetection.logger import logging
from dotenv import load_dotenv
import pickle

load_dotenv()

url=os.getenv("MONGO_DB_URL")
db=os.getenv("db_name")
coll=os.getenv("collection_name")

def export_collection_as_dataframe(url,coll, db):
    try:
        logging.info("started connecting",)
        
        mongo_client = MongoClient(url)
        collection = mongo_client[db][coll]
        df = pd.DataFrame(list(collection.find()))
        
        if "_id" in df.columns.to_list():
            df = df.drop(columns=["_id"], axis=1)

        df.replace({"na": np.nan}, inplace=True)
        logging.info("connection established",)
        print(df.head())
        return df
    
    except Exception as e:
        raise CustomException(e, sys)


def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)







def load_object(file_path):
    try:
        with open(file_path, "rb") as file_obj:
            return dill.load(file_obj)

    except Exception as e:
        raise CustomException(e, sys)   