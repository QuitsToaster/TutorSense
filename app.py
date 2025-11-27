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
def dashboard():  # make dashboard the landing page
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
    else:
        total_students = passing = at_risk = average_gpa = 0

    return render_template("dashboard.html",
                           stats={'total_students': total_students, 'passing': passing, 'at_risk': at_risk, 'average_gpa': average_gpa},
                           active_page='dashboard')

@app.route("/predict", methods=["GET", "POST"])
def index():
    global student_data
    prediction = None
    selected_student = None

    if student_data.empty:
        students_list = []
    else:
        students_list = student_data['StudentID'].tolist()

    if request.method == "POST":
        student_id = int(request.form["student_id"])
        selected_student = student_data[student_data['StudentID'] == student_id].iloc[0]

        features = [[
            selected_student['StudentID'],
            selected_student['Attendance'],
            selected_student['HomeworkRate'],
            selected_student['MidtermScore'],
            selected_student['Participation'],
            selected_student['PreviousGPA']
        ]]
        pred = model.predict(features)[0]
        prediction = "PASS" if pred == 1 else "FAIL — Needs Tutoring"

    return render_template("index.html",
                           prediction=prediction,
                           students_list=students_list,
                           selected_student_id=selected_student['StudentID'] if selected_student is not None else None,
                           active_page='predict')

@app.route("/about")
def about():
    return render_template("about.html", active_page='about')

if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
