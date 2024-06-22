from flask import Flask, render_template, request
import new  # Assuming the new.py module is in the same directory

app = Flask(__name__)

@app.route("/", methods=["GET", "POST"])
def hello():
    disorder = None
    if request.method == "POST":
        gender = request.form['gender']
        age = int(request.form['age'])
        sleep_duration = float(request.form['sd'])
        quality_of_sleep = int(request.form['qos'])
        physical_activity = int(request.form['physical_activity'])
        stress = int(request.form['stress'])
        bmi = request.form['bmi']
        blood_pressure = request.form['Blood_Pressure']
        heart_rate = int(request.form['Heart_Rate'])
        daily_steps = int(request.form['Daily_Steps'])
        
        disorder = new.predict_disorder(
            gender, age, sleep_duration, quality_of_sleep, physical_activity, stress, bmi, blood_pressure, heart_rate, daily_steps
        )
    
    return render_template("index.html", disorder=disorder)

if __name__ == "__main__":
    app.run(debug=True)
