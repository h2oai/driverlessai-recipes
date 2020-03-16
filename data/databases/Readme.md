# Driverless AI Recipe: Create a Dataset from a MongoDB Collection 

This recipe is intended to be the starting point to import a dataset from MongoDB. 

## Please fill before usage

To connect to your MongoDB collection, fill the connection string
 ```
 MONGO_CONNECTION_STRING = "mongodb+srv://<username>:<password>@host[/[database][?options]]"
 ```

Notes:
* This information is logged in Driverless AI logs.
* This recipe transform all Object types to string. Data type can be changed using: 
```
df["column_name"] = df["column_name"].astype(type)
``` 