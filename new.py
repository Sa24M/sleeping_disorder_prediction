import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import joblib
from flask import Flask, render_template, request

app = Flask(__name__)

# Load and preprocess the dataset
def load_and_preprocess_data():
    df = pd.read_csv("sleeping_disorder_prediction/sleephealth.csv")
    df['Sleep Disorder'].fillna('no disorder', inplace=True)
    df = df.drop(['Occupation', 'Person ID'], axis=1)

    gender_encoder = LabelEncoder()
    bmi_encoder = LabelEncoder()
    sleep_disorder_encoder = LabelEncoder()

    df['Gender'] = gender_encoder.fit_transform(df['Gender'])
    df['BMI Category'] = bmi_encoder.fit_transform(df['BMI Category'])
    df['Sleep Disorder'] = sleep_disorder_encoder.fit_transform(df['Sleep Disorder'])

    df[['Systolic', 'Diastolic']] = df['Blood Pressure'].str.split('/', expand=True)
    df['Systolic'] = pd.to_numeric(df['Systolic'])
    df['Diastolic'] = pd.to_numeric(df['Diastolic'])

    df.drop(columns=['Blood Pressure'], inplace=True)

    return df, gender_encoder, bmi_encoder, sleep_disorder_encoder

# Train the model and save it along with encoders
def train_model():
    df, gender_encoder, bmi_encoder, sleep_disorder_encoder = load_and_preprocess_data()
    x = df.drop(['Sleep Disorder'], axis=1)
    y = df['Sleep Disorder']

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    accuracy = model.score(X_test, y_test)
    print("Accuracy is ", accuracy)

    # Save the trained model and encoders
    joblib.dump(model, 'model.pkl')
    joblib.dump(gender_encoder, 'gender_encoder.pkl')
    joblib.dump(bmi_encoder, 'bmi_encoder.pkl')
    joblib.dump(sleep_disorder_encoder, 'sleep_disorder_encoder.pkl')

    return model

# Load the model and encoders from file
def load_model():
    model = joblib.load('model.pkl')
    gender_encoder = joblib.load('gender_encoder.pkl')
    bmi_encoder = joblib.load('bmi_encoder.pkl')
    sleep_disorder_encoder = joblib.load('sleep_disorder_encoder.pkl')
    return model, gender_encoder, bmi_encoder, sleep_disorder_encoder

# Function to predict disorder
def predict_disorder(gender, age, sleep_duration, quality_of_sleep, physical_activity, stress, bmi, blood_pressure, heart_rate, daily_steps):
    # Load the trained model and encoders
    model, gender_encoder, bmi_encoder, sleep_disorder_encoder = load_model()
    
    # Preprocess input data
    systolic, diastolic = map(int, blood_pressure.split('/'))
    gender_encoded = gender_encoder.transform([gender])[0]
    bmi_encoded = bmi_encoder.transform([bmi])[0]
    
    # Reshape the input into a 2D array with one row
    prediction_data = np.array([gender_encoded, age, sleep_duration, quality_of_sleep, physical_activity, stress, bmi_encoded, systolic, diastolic, heart_rate, daily_steps]).reshape(1, -1)
    
    # Make prediction
    prediction = model.predict(prediction_data)
    
    # Decode the prediction
    prediction_decoded = sleep_disorder_encoder.inverse_transform(prediction)
    return prediction_decoded[0]

# Ensure the model is trained and saved when the script is run
if __name__ == "__main__":
    train_model()
