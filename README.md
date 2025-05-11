
# 💤 Sleeping Disorder Prediction Web App

This is a Flask-based web application that predicts the type of sleep disorder (if any) based on user inputs such as gender, age, sleep duration, stress levels, and more. The prediction is powered by a trained Random Forest model.

---

## 📁 Project Structure

```
.
├── app.py                     # Flask application – handles routes and rendering
├── new.py                     # ML model loading and prediction logic
├── model.pkl                  # Trained Random Forest model
├── gender_encoder.pkl         # Label encoder for 'Gender'
├── bmi_encoder.pkl            # Label encoder for 'BMI Category'
├── sleep_disorder_encoder.pkl # Label encoder for target 'Sleep Disorder'
├── templates/
│   └── index.html             # HTML form for input and displaying results
├── sleeping_disorder_prediction/
│   └── sleephealth.csv        # Original dataset (used for training)
```

---

## 🚀 Features

- User-friendly web form to input personal and health data.
- Real-time prediction of possible sleeping disorders.
- Displays one of the following results:
  - No Disorder
  - Insomnia
  - Sleep Apnea

---


**requirements.txt**
```
pandas
numpy
scikit-learn
joblib
flask
```

### 4. Train the Model (Optional)

If you want to retrain the model using the CSV data:

```bash
python new.py
```

This will regenerate:
- `model.pkl`
- `gender_encoder.pkl`
- `bmi_encoder.pkl`
- `sleep_disorder_encoder.pkl`

### 5. Run the Flask App

```bash
python app.py
```

Then open your browser and go to:
```
http://127.0.0.1:5000/
```

---


## 🧠 Model Info

- **Algorithm**: Random Forest Classifier
- **Dataset**: `sleephealth.csv` (with health and lifestyle metrics)
- **Target**: Sleep Disorder (Categorical)

---

## 📂 Data Fields (User Inputs)

- Gender
- Age
- Sleep Duration (hours)
- Quality of Sleep (1–10 scale)
- Physical Activity Level
- Stress Level
- BMI Category
- Blood Pressure (e.g., 120/80)
- Heart Rate
- Daily Steps

---


## 👩‍💻 Author

- Sakshi Mishra, CST Dept., IIEST Shibpur
