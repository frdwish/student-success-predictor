# Student Success Predictor & Analytics Dashboard

Predict student final grades using machine learning, with a fully interactive dashboard for visualization and exploration.

---

## Project Overview

This project leverages the UCI Student Performance dataset to create an end-to-end machine learning pipeline. It combines:

* **Exploratory Data Analysis (EDA):** Understand trends, distributions, and correlations.
* **Data Cleaning & Preprocessing:** Handle missing values, encode categorical features, remove outliers.
* **Feature Engineering:** Select the most impactful features like study time, past grades (G1, G2), absences, parental education, etc.
* **Machine Learning Model:** RandomForestRegressor predicts the final grade (G3).
* **Streamlit Web App:** Interactive UI for predicting grades and exploring dashboards with charts and evaluation metrics.

---

## Key Features

* Predict **final grade (G3)** for a student based on study and performance factors.
* **Visual analytics dashboard** with:

  * Histograms and KDE plots of grades
  * Boxplots of numeric features vs final grade
  * Scatter plots (study time, absences, failures vs final grade)
  * Correlation heatmap
* Model evaluation metrics: **MAE, RMSE, R²**
* Final grade scaled to **0–100** for better visualization.

---

## Project Structure

```
student-success-predictor/
│── data/                  # Raw and cleaned dataset
│     └── cleaned_student-mat.csv
│── notebooks/             # EDA + ML training scripts
│     └── 01_EDA_and_Modeling.py
│── model/                 # Trained ML model
│     └── student_model.pkl
│── app/                   # Streamlit Web App
│     └── streamlit_app.py
│── reports/               # VTU report / case study
│     └── VTU_Project_Report.docx
├── requirements.txt       # All Python dependencies
└── README.md              # Project overview and instructions
```

---

## Setup & Installation

1. **Clone the repository**

```bash
git clone https://github.com/<your-username>/student-success-predictor.git
cd student-success-predictor
```

2. **Create a virtual environment** (recommended)

```bash
python3 -m venv venv
source venv/bin/activate      # Linux/Mac
venv\Scripts\activate       # Windows
```

3. **Install dependencies**

```bash
pip install -r requirements.txt
```

---

## Running the Project

###  Train the Model & Clean Data

```bash
python notebooks/01_EDA_and_Modeling.py
```

* Cleans the dataset → `data/cleaned_student-mat.csv`
* Trains RandomForest model → `model/student_model.pkl`

###  Run the Streamlit Web App

```bash
streamlit run app/streamlit_app.py
```

* Interactive sidebar for student inputs
* Tabs for dataset preview, analytics dashboard, scatter plots, and model evaluation
* Predicts final grade (0–100) instantly

---

## Model Output

| File                           | Description                                   |
| ------------------------------ | --------------------------------------------- |
| `data/cleaned_student-mat.csv` | Preprocessed dataset ready for analysis       |
| `model/student_model.pkl`      | Trained RandomForestRegressor for predictions |

---

## Technologies Used

* Python, Pandas, NumPy
* Scikit-Learn (RandomForest)
* Matplotlib & Seaborn (visualizations)
* Streamlit (interactive web dashboard)

---

## Future Enhancements

* Include additional features like attendance trends and extra-curricular activities
* Deploy dashboard on **AWS / Heroku / Streamlit Cloud** for global access
* Add **feature importance charts** for interpretability

---

## Conclusion

This project demonstrates a **complete ML workflow**: data exploration, cleaning, modeling, evaluation, and deployment. It provides **predictive insights** into student performance and an **interactive dashboard** for analysis.
