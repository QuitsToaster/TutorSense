from flask import Flask, render_template, request
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'


model = joblib.load('tutoring_model.pkl')


student_data = pd.DataFrame(columns=[
    'StudentID','Attendance','HomeworkRate','MidtermScore',
    'Participation','PreviousGPA'
])


def generate_graph(title, labels, values, ymin=0, ymax=100):
    plt.figure(figsize=(4,3))
    plt.bar(labels, values)
    plt.title(title)
    plt.ylim(ymin, ymax)

    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', bbox_inches="tight")
    buffer.seek(0)
    graph_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()

    return graph_base64


@app.route("/", methods=["GET", "POST"])
def dashboard():
    global student_data


    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename.endswith(".csv"):
            df = pd.read_csv(file)
            student_data = df  


    if student_data.empty:
        return render_template(
            "dashboard.html",
            stats={
                'total_students': 0,
                'passing': 0,
                'at_risk': 0,
                'average_gpa': 0,
                'literacy_risk_pct': 0,
                'average_predicted_grade': 0,
                'performance_insight': "No data available.",
                'recommendations': [],
                'graph_literacy': None,
                'graph_grade': None
            },
            active_page='dashboard'
        )

    total_students = len(student_data)

    features = student_data[
        ['StudentID','Attendance','HomeworkRate','MidtermScore','Participation','PreviousGPA']
    ]

    predictions = model.predict(features)

    passing = int((predictions == 1).sum())
    at_risk = int((predictions == 0).sum())

    literacy_risk_pct = round((at_risk / total_students) * 100, 2)
    average_gpa = round(float(student_data['PreviousGPA'].mean()), 2)

    try:
        predicted_probs = model.predict_proba(features)[:, 1] * 100
        average_predicted_grade = round(predicted_probs.mean(), 2)
    except:
        average_predicted_grade = round(student_data['MidtermScore'].mean(), 2)

    if literacy_risk_pct > 50:
        performance_insight = "High number of students are at risk. Immediate intervention required."
    elif literacy_risk_pct > 25:
        performance_insight = "Moderate risk level. Some students need targeted support."
    else:
        performance_insight = "Overall performance looks healthy."

    recommendations = []
    if literacy_risk_pct > 40:
        recommendations.append("Increase tutoring sessions and reading reinforcement.")
    if average_predicted_grade < 75:
        recommendations.append("Review instructional pacing and difficulty.")
    if average_gpa < 2.5:
        recommendations.append("Provide academic advising to boost GPA trends.")
    if not recommendations:
        recommendations.append("Students are performing well. Keep current strategy.")

    graph_literacy = generate_graph(
        "Average Literacy Risk (%)",
        ["Literacy Risk %"],
        [literacy_risk_pct],
        ymin=0, ymax=100
    )

    graph_grade = generate_graph(
        "Predicted Average Final Grade",
        ["Predicted Grade"],
        [average_predicted_grade],
        ymin=0, ymax=100
    )

    return render_template(
        "dashboard.html",
        stats={
            'total_students': total_students,
            'passing': passing,
            'at_risk': at_risk,
            'average_gpa': average_gpa,
            'literacy_risk_pct': literacy_risk_pct,
            'average_predicted_grade': average_predicted_grade,
            'performance_insight': performance_insight,
            'recommendations': recommendations,
            'graph_literacy': graph_literacy,
            'graph_grade': graph_grade
        },
        active_page='dashboard'
    )


# -------------------------------------------------
# STUDENT PREDICTOR PAGE
# -------------------------------------------------
@app.route("/predict", methods=["GET", "POST"])
def index():
    global student_data
    prediction = None
    selected_student = None

    students_list = student_data['StudentID'].tolist() if not student_data.empty else []

    if request.method == "POST":
        student_id = int(request.form["student_id"])
        selected_student = student_data.loc[student_data['StudentID'] == student_id].iloc[0]

        features = [[
            selected_student['StudentID'],
            selected_student['Attendance'],
            selected_student['HomeworkRate'],
            selected_student['MidtermScore'],
            selected_student['Participation'],
            selected_student['PreviousGPA']
        ]]

        raw_pred = model.predict(features)[0]

        risk_level = "Low" if raw_pred == 1 else "High"
        literacy_score = int(max(0, min(100, selected_student['HomeworkRate'] * 10)))
        predicted_grade = int(selected_student['PreviousGPA'] * 25)

        reason = "Good performance indicators." if raw_pred == 1 else "Indicators suggest academic risk."
        recommendation = "Maintain study habits." if raw_pred == 1 else "Recommend tutoring support."

        prediction = {
            "risk_level": risk_level,
            "literacy_score": literacy_score,
            "predicted_grade": predicted_grade,
            "reason": reason,
            "recommendation": recommendation
        }

    return render_template(
        "index.html",
        prediction=prediction,
        students_list=students_list,
        selected_student_id=selected_student['StudentID'] if selected_student is not None else None,
        active_page='predict'
    )


@app.route("/about")
def about():
    return render_template("about.html", active_page='about')


if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
