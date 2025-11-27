# TutorSense – Student Risk Prediction System

TutorSense is an AI-powered web application that predicts which students are at risk of failing a course. Built using **Python, Flask, Tailwind CSS**, and **Random Forest Machine Learning**, it helps teachers and schools identify at-risk students early so they can provide support.

## **Features**

* Predict if a single student will pass or fail a course
* Upload CSV files with multiple students’ data
* View dashboard with statistics: total students, passing students, at-risk students, and average GPA
* Modern responsive UI using Tailwind CSS
* Fully functional backend using Flask

## **System Requirements**

* Python 3.9+
* pip
* Flask
* pandas
* scikit-learn
* joblib
* Tailwind CSS (loaded via CDN)

## **Project Structure**

```
TutorSense/
│── app.py
│── tutoring_model.pkl
│── templates/
│     ├── base.html
│     ├── index.html
│     ├── dashboard.html
│     └── about.html
│── uploads/

```

## **Installation & Setup**

1. **Clone the repository**

```bash
git clone https://github.com/QuitsToaster/TutorSense
cd TutorSense
```

2. **Create a virtual environment (optional but recommended)**

```bash
python3 -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

4. **Ensure your model is saved as `tutoring_model.pkl`** in the project root. If not, train your Random Forest model and save it using:

```python
import joblib
joblib.dump(model, 'tutoring_model.pkl')
```

5. **Run the Flask app**

```bash
python app.py (Windows)
or
python3 app.py (Mac)
```

6. **Open your browser** and go to:

```
http://127.0.0.1:5000/
```

---

## **How the System Works**

1. **Predictor Page (`/`)**

   * Enter a student's details: Student ID, Attendance, Homework Rate, Midterm Score, Participation, and Previous GPA.
   * Click **Predict** to see if the student is predicted to **PASS** or **FAIL**.

2. **Dashboard Page (`/dashboard`)**

   * Displays total students, passing students, at-risk students, and average GPA.
   * Upload a **CSV file** with multiple students’ data to update the dashboard statistics.
   * CSV file must have columns:

     ```
     StudentID, Attendance, HomeworkRate, MidtermScore, Participation, PreviousGPA
     ```

3. **About Page (`/about`)**

   * Explains the purpose of the system and technologies used.

## **CSV Upload Format Example**

```csv
StudentID,Attendance,HomeworkRate,MidtermScore,Participation,PreviousGPA
101,1,1,88,1,3.5
102,0,1,70,0,2.8
103,1,0,60,1,3.0
```

## **Technologies Used**

* **Python** – backend
* **Flask** – web framework
* **scikit-learn** – machine learning (Random Forest)
* **pandas** – data processing
* **joblib** – saving/loading model
* **Tailwind CSS** – styling
* **HTML & Jinja2** – frontend templates


Do you want me to make that too?
