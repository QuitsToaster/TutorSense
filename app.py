from flask import Flask, render_template, request, jsonify, url_for, redirect
import joblib
import pandas as pd
import numpy as np
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io, base64, json

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'

model = joblib.load('tutoring_model.pkl')

student_data = pd.DataFrame(columns=[
    'StudentID','Attendance','HomeworkRate','MidtermScore',
    'Participation','PreviousGPA','FinalScore'
])


# ---------------------------
# Helper: Matplotlib -> base64
# ---------------------------
def fig_to_base64(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', bbox_inches="tight")
    buf.seek(0)
    data = base64.b64encode(buf.getvalue()).decode('utf-8')
    plt.close(fig)
    return data

def generate_line_trend(student_row):
    mid = float(student_row.get('MidtermScore', 0))
    final = float(student_row.get('FinalScore', np.nan) if 'FinalScore' in student_row.index else np.nan)
    
    scores = [mid, final if not pd.isna(final) else mid]
    labels = ['Midterm', 'Final']
    fig, ax = plt.subplots(figsize=(8, 3.5))
    ax.plot(labels, scores, marker='o', linewidth=2)
    ax.set_ylim(0, 100)
    ax.set_title(f"Score Trend (Student {student_row['StudentID']})")
    ax.set_ylabel("Score")
    ax.grid(True, linestyle='--', alpha=0.4)
    return fig_to_base64(fig)

def generate_metrics_bar(student_row):
    
    attendance = float(student_row.get('Attendance', 0)) * 100
    homework = float(student_row.get('HomeworkRate', 0)) * 100
    participation = float(student_row.get('Participation', 0)) * 100
   
    prev_gpa = float(student_row.get('PreviousGPA', 0))
    if prev_gpa <= 6:  
        prev_gpa_scaled = (prev_gpa / 6.0) * 100
    else:
        prev_gpa_scaled = min(prev_gpa, 100)

    labels = ['Attendance', 'Homework', 'Participation', 'PreviousGPA']
    values = [attendance, homework, participation, prev_gpa_scaled]

    fig, ax = plt.subplots(figsize=(8, 3.5))
    bars = ax.bar(labels, values)
    ax.set_ylim(0, 100)
    ax.set_title(f"Key Metrics (Student {student_row['StudentID']})")
    ax.set_ylabel("Value (0-100 scaled)")
    ax.grid(axis='y', linestyle='--', alpha=0.3)
    
    for bar, val in zip(bars, values):
        ax.text(bar.get_x() + bar.get_width() / 2, val + 1, f"{int(round(val))}", ha='center', va='bottom', fontsize=9)
    return fig_to_base64(fig)


# ---------------------------------------------
# DASHBOARD ROUTE 
# ---------------------------------------------
@app.route("/", methods=["GET", "POST"])
def dashboard():
    global student_data

    if request.method == "POST":
        file = request.files.get("file")
        if file and file.filename.endswith(".csv"):
            df = pd.read_csv(file)
            
            df_cols = [c.strip() for c in df.columns]
            df.columns = df_cols
            
            col_map = {}
            for c in df.columns:
                lc = c.lower()
                if lc in ['studentid', 'student_id', 'id']:
                    col_map[c] = 'StudentID'
                elif lc in ['attendance']:
                    col_map[c] = 'Attendance'
                elif lc in ['homeworkrate','homework_rate','homework']:
                    col_map[c] = 'HomeworkRate'
                elif lc in ['midtermscore','midterm_score','midterm']:
                    col_map[c] = 'MidtermScore'
                elif lc in ['participation']:
                    col_map[c] = 'Participation'
                elif lc in ['previousgpa','previous_gpa','gpa','previous']:
                    col_map[c] = 'PreviousGPA'
                elif lc in ['finalscore','final_score','final']:
                    col_map[c] = 'FinalScore'
            df = df.rename(columns=col_map)
            
            for col in ['StudentID','Attendance','HomeworkRate','MidtermScore','Participation','PreviousGPA','FinalScore']:
                if col not in df.columns:
                    df[col] = np.nan
            student_data = df[['StudentID','Attendance','HomeworkRate','MidtermScore','Participation','PreviousGPA','FinalScore']].copy()

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
    features = student_data[['StudentID','Attendance','HomeworkRate','MidtermScore','Participation','PreviousGPA']].fillna(0)
    
    try:
        predictions = model.predict(features)
    except Exception:
        predictions = np.zeros(total_students, dtype=int)

    passing = int((predictions == 1).sum())
    at_risk = int((predictions == 0).sum())
    literacy_risk_pct = round((at_risk / total_students) * 100, 2)
    average_gpa = round(float(student_data['PreviousGPA'].astype(float).mean()), 2)

    try:
        predicted_probs = model.predict_proba(features)[:, 1] * 100
        average_predicted_grade = round(predicted_probs.mean(), 2)
    except:
        average_predicted_grade = round(student_data['MidtermScore'].astype(float).mean(), 2)

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

    # -----------------------
    # Modern Graph Generator
    # -----------------------
    def generate_graph(title, labels, values, color="#4f46e5", ymin=0, ymax=100):
        fig, ax = plt.subplots(figsize=(5,3))
        bars = ax.bar(labels, values, color=color, edgecolor='none', width=0.5)
        
        # Gradient effect (approximate using alpha)
        for bar in bars:
            bar.set_alpha(0.85)
            bar.set_linewidth(0)
            bar.set_edgecolor("none")
            bar.set_capstyle("round")
        
        # Remove top and right spines
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.spines['left'].set_color("#cbd5e1")
        ax.spines['bottom'].set_color("#cbd5e1")
        
        # Subtle grid
        ax.yaxis.grid(True, linestyle='--', alpha=0.3)
        ax.set_axisbelow(True)
        
        # Bar labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, height + 1, f"{height:.1f}", ha='center', va='bottom', fontsize=10, fontweight='bold', color="#1e293b")
        
        ax.set_ylim(ymin, ymax)
        ax.set_title(title, fontsize=12, fontweight='bold', pad=10, color="#1e293b")
        ax.set_facecolor("#f1f5f9")
        fig.patch.set_facecolor("#f8fafc")
        
        plt.tight_layout()
        return fig_to_base64(fig)

    graph_literacy = generate_graph("Average Literacy Risk (%)", ["Literacy Risk %"], [literacy_risk_pct], color="#22c55e", ymin=0, ymax=100)
    graph_grade = generate_graph("Predicted Average Final Grade", ["Predicted Grade"], [average_predicted_grade], color="#3b82f6", ymin=0, ymax=100)

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



# ---------------------------------------------
# PREDICTOR PAGE: list students paginated + charts per student
# ---------------------------------------------
@app.route("/predict", methods=["GET", "POST"])
def index():
    global student_data

    try:
        page = int(request.args.get('page', '1'))
        if page < 1:
            page = 1
    except:
        page = 1
    per_page = 10

    if student_data.empty:
        students_list = []
        student_charts = {}
    else:
        
        sd = student_data.copy()
        sd['StudentID'] = sd['StudentID'].astype(str)
        
        students_all = sd.to_dict(orient='records')
        total = len(students_all)
        start = (page - 1) * per_page
        end = start + per_page
        students_page = students_all[start:end]

        student_charts = {}
        for s in students_page:
            sid = s['StudentID']
            
            row = pd.Series({
                'StudentID': s.get('StudentID'),
                'Attendance': s.get('Attendance', 0),
                'HomeworkRate': s.get('HomeworkRate', 0),
                'MidtermScore': s.get('MidtermScore', 0),
                'Participation': s.get('Participation', 0),
                'PreviousGPA': s.get('PreviousGPA', 0),
                'FinalScore': s.get('FinalScore', np.nan)
            })
            try:
                line_b64 = generate_line_trend(row)
                bar_b64 = generate_metrics_bar(row)
            except Exception as e:
                
                line_b64 = ""
                bar_b64 = ""
            student_charts[sid] = {
                'line': line_b64,
                'bars': bar_b64,
                'data': s
            }

        students_list = students_page
        total_pages = (total + per_page - 1) // per_page

    return render_template(
        "index.html",
        students=students_list,
        charts=student_charts,
        page=page,
        per_page=per_page,
        total_pages=total_pages if not student_data.empty else 0,
        active_page='predict'
    )


# ---------------------------------------------
# AJAX: Predict single student 
# ---------------------------------------------
@app.route("/predict_student", methods=["POST"])
def predict_student():
    global student_data
    payload = request.get_json() or {}
    student_id = str(payload.get('student_id', ''))
    if student_data.empty or student_id == '':
        return jsonify({"error": "No student data available or invalid id"}), 400

    df = student_data.copy()
    df['StudentID'] = df['StudentID'].astype(str)
    match = df.loc[df['StudentID'] == student_id]
    if match.empty:
        return jsonify({"error": "Student not found"}), 404

    row = match.iloc[0]

    features = np.array([[
        row['StudentID'] if not pd.isna(row['StudentID']) else 0,
        row['Attendance'] if not pd.isna(row['Attendance']) else 0,
        row['HomeworkRate'] if not pd.isna(row['HomeworkRate']) else 0,
        row['MidtermScore'] if not pd.isna(row['MidtermScore']) else 0,
        row['Participation'] if not pd.isna(row['Participation']) else 0,
        row['PreviousGPA'] if not pd.isna(row['PreviousGPA']) else 0
    ]])
    
    try:
        raw_pred = model.predict(features)[0]
        prob = None
        try:
            prob = float(model.predict_proba(features)[0][1]) * 100
        except:
            prob = None
    except Exception:
        raw_pred = 0
        prob = None

    risk_level = "Low" if raw_pred == 1 else "High"
    literacy_score = int(max(0, min(100, (row.get('HomeworkRate', 0) or 0) * 100)))
    predicted_grade = int((row.get('PreviousGPA', 0) or 0) * 25)  

    reason = "Good performance indicators." if raw_pred == 1 else "Indicators suggest academic risk."
    recommendation = "Maintain study habits." if raw_pred == 1 else "Recommend tutoring support."

    result = {
        "student_id": student_id,
        "risk_level": risk_level,
        "literacy_score": literacy_score,
        "predicted_grade": predicted_grade,
        "probability": prob,
        "reason": reason,
        "recommendation": recommendation
    }
    return jsonify(result)


# ---------------------------------------------
# ABOUT PAGE 
# ---------------------------------------------
@app.route("/about")
def about():
    team = [
        {"name": "Bryan Lloyd T. Tan", "role": "Lead Developer", "image": "team/member1.jpg"},
        {"name": "Timothy John M. Lardizabal", "role": "Backend Developer", "image": "team/member2.jpg"},
        {"name": "Arem A. Ancheta", "role": "Frontend Developer", "image": "team/member3.jpg"},
        {"name": "Aj Karl Ancheta", "role": "Frontend Developer", "image": "team/member4.jpg"},
        {"name": "Chelsea Leigh M. Pascua", "role": "Project Manager I", "image": "team/member5.jpg"},
        {"name": "Trishea Andrea A. Liwanag", "role": "Project Manager II", "image": "team/member6.jpg"},
    ]
    return render_template(
        "about.html",
        active_page='about',
        team=team
    )

# ---------------------------------------------
# VIEW PASSING / AT-RISK STUDENTS 
# ---------------------------------------------
@app.route("/students/<group>")
def view_students_group(group):
    global student_data

    page = request.args.get("page", 1, type=int)
    per_page = 10  

    if student_data.empty:
        return render_template(
            "student_list.html",
            title="No Data Available",
            students=[],
            page=1,
            total_pages=1,
            group=group
        )

    features = student_data[['StudentID','Attendance','HomeworkRate','MidtermScore','Participation','PreviousGPA']].fillna(0)
    try:
        preds = model.predict(features)
    except:
        preds = np.zeros(len(student_data), dtype=int)

    df = student_data.copy()
    df['Prediction'] = preds

    if group == "passing":
        filtered = df[df['Prediction'] == 1]
        title = "Passing Students"
    elif group == "at-risk":
        filtered = df[df['Prediction'] == 0]
        title = "At-Risk Students"
    else:
        filtered = df
        title = "Students"

    students_list = filtered.to_dict(orient='records')

    total_students = len(students_list)
    total_pages = max(1, (total_students + per_page - 1) // per_page)

    start = (page - 1) * per_page
    end = start + per_page
    page_students = students_list[start:end]

    return render_template(
        "student_list.html",
        title=title,
        students=page_students,
        page=page,
        total_pages=total_pages,
        group=group
    )


# ---------------------------------------------
# RUN APP
# ---------------------------------------------
if __name__ == '__main__':
    if not os.path.exists('uploads'):
        os.makedirs('uploads')
    app.run(debug=True)
