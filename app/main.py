#!/usr/bin/env python
# coding: utf-8

# ## Deployment (FastAPI)

# In[42]:


import os
print("Current working directory:")
os.getcwd()


# In[44]:


print("Files in models folder:")
os.listdir("C:\\Users\\shrav\\Bank_Churn_Prediction\\models")


# In[39]:


from fastapi import FastAPI
import joblib
import pandas as pd

# Initialize app
app = FastAPI()

# Load model
model = joblib.load(
    "C:/Users/shrav/Bank_Churn_Prediction/models/xgboost_model.pkl"
)

# Home route

@app.get("/")
def home():

    return {

        "message":
        "Bank Customer Churn Prediction API Running"

    }

# Preprocessing (must match notebook)

def preprocess_input(data):

    df = pd.DataFrame([data])

    # Encode Gender
    df["Gender"] = df["Gender"].map({
        "Female": 0,
        "Male": 1
    })

    # One-hot Geography
    df["Geography_Germany"] = (
        df["Geography"] == "Germany"
    ).astype(int)

    df["Geography_Spain"] = (
        df["Geography"] == "Spain"
    ).astype(int)

    df.drop(
        "Geography",
        axis=1,
        inplace=True
    )

    # Feature Engineering

    df["BalanceSalaryRatio"] = (
        df["Balance"] /
        (df["EstimatedSalary"] + 1)
    )

    df["TenureAgeRatio"] = (
        df["Tenure"] /
        (df["Age"] + 1)
    )

    return df


# Prediction endpoint

@app.post("/predict")

def predict(data: dict):

    processed = preprocess_input(data)

    prediction = model.predict(
        processed
    )[0]

    probability = model.predict_proba(
        processed
    )[0][1]

    return {

        "churn_prediction":
        int(prediction),

        "churn_probability":
        float(probability)

    }


# In[ ]:





# In[ ]:




