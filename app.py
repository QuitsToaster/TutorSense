from flask import Flask, render_template, request, redirect, url_for
import joblib
import numpy as np
import pandas as pd
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

# Load model
model = joblib.load('tutoring_model.pkl')

# Initialize empty student data
student_data = pd.DataFrame(columns=['StudentID','Attendance','HomeworkRate','MidtermScore','Participation','PreviousGPA'])

@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    if request.method == "POST":
        student_id = int(request.form["student_id"])
        attendance = float(request.form["attendance"])
        homework = float(request.form["homework"])
        midterm = float(request.form["midterm"])
        participation = float(request.form["participation"])
        gpa = float(request.form["gpa"])

        features = [[student_id, attendance, homework, midterm, participation, gpa]]
        pred = model.predict(features)[0]
        prediction = "PASS" if pred == 1 else "FAIL — Needs Tutoring"

    return render_template("index.html", prediction=prediction)

@app.route("/dashboard", methods=["GET", "POST"])
def dashboard():
    global student_data

    if request.method == "POST":
        # CSV Upload
        file = request.files.get("file")
        if file and file.filename.endswith(".csv"):
            df = pd.read_csv(file)
            student_data = df  # overwrite current data

    # Compute stats
    if not student_data.empty:
        total_students = len(student_data)
        features = student_data[['StudentID','Attendance','HomeworkRate','MidtermScore','Participation','PreviousGPA']]
        predictions = model.predict(features)
        passing = int((predictions == 1).sum())
        at_risk = int((predictions == 0).sum())
        average_gpa = float(student_data['PreviousGPA'].mean())
        chart_data = predictions.astype(int).tolist()  # ensure it's a list of ints
    else:
        total_students = passing = at_risk = average_gpa = 0
        chart_data = [0, 0]  # default values for chart

    return render_template("dashboard.html",
                           stats={'total_students': total_students, 'passing': passing, 'at_risk': at_risk, 'average_gpa': average_gpa},
                           chart_data=chart_data)

@app.route("/about")
def about():
    return render_template("about.html")

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
