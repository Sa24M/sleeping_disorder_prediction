import pandas as pd
import numpy as np 
import seaborn as sb
df=pd.read_csv("sleephealth.csv")
df['Sleep Disorder'].fillna('no disorder', inplace=True)
df = df.drop(['Occupation','Person ID'], axis=1)
from sklearn.preprocessing import LabelEncoder
label_encoder = LabelEncoder()
df['Gender'] = label_encoder.fit_transform(df['Gender'])
df['BMI Category'] = label_encoder.fit_transform(df['BMI Category'])
df['Sleep Disorder'] = label_encoder.fit_transform(df['Sleep Disorder'])
df['Blood Pressure'] = label_encoder.fit_transform(df['Blood Pressure'])
x=df.drop(['Sleep Disorder'],axis=1)
y=df['Sleep Disorder']
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=20)
model = RandomForestClassifier()
model.fit(X_train, y_train)
from sklearn.metrics import accuracy_score
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("accuracy is ",accuracy)
