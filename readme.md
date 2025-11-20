 # Student Success Predictor & Analytics Dashboard---->

This project uses the UCI Student Performance dataset to build:
- Full EDA (Exploratory Data Analysis)
- Data Cleaning & Preprocessing
- Feature Engineering
- Machine Learning Model (RandomForest)
- Streamlit Web App for predictions + dashboard

## Project Structure
student-success-predictor/
│── data/
│── notebooks/
│── model/
│── app/
│── reports/

## How to Run
1. Install dependencies:
   pip install -r requirements.txt

2. Run the EDA + ML training notebook/code:
   python notebooks/01_EDA_and_Modeling.py

3. Run the Streamlit App:
   streamlit run app/streamlit_app.py

## Model Output
- Cleaned dataset saved in data/cleaned_student-mat.csv
- Trained model saved in model/student_model.pkl
