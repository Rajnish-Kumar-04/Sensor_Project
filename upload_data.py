from pymongo.mongo_client import MongoClient
import pandas as pd
import json

#url
uri = "mongodb+srv://bt25csa043_db_user:RajnishKumar2007@cluster0.vmgd0ax.mongodb.net/?appName=Cluster0"

#create a new client and connect to server
client = MongoClient(uri)

#create a database name and collection name
Database_Name = 'Detection'
Collection_Name = 'waferfault'

df = pd.read_csv("C:\Users\Rajnish\Machine Learning\Detection_Project\notebooks\wafer_23012020_041211.csv")

df.drop("Unnamed: 0",axis = 1)

json_record = list(json.loads(df.T.to_json()).values())

#Push json data in monga db
client[Database_Name][Collection_Name].insert_many(json_record)