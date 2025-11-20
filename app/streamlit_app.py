import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt

st.set_page_config(page_title="Student Success Predictor", layout="wide")
sns.set_style("whitegrid")

# Load model
model = joblib.load("model/student_model.pkl")

st.title("🏫 Student Success Predictor & Analytics Dashboard ")

# Sidebar Inputs
st.sidebar.header("Input Parameters for Prediction")

col1, col2 = st.sidebar.columns(2)
studytime = col1.slider("Study Time (1–4)", 1, 4, 2)
traveltime = col2.slider("Travel Time (1–4)", 1, 4, 1)
failures = st.sidebar.slider("Past Failures", 0, 4, 0)
absences = st.sidebar.slider("Absences", 0, 50, 5)
G1 = st.sidebar.slider("G1 (0–20)", 0, 20, 10)
G2 = st.sidebar.slider("G2 (0–20)", 0, 20, 12)
col3, col4 = st.sidebar.columns(2)
Medu = col3.slider("Mother's Education (0–4)", 0, 4, 2)
Fedu = col4.slider("Father's Education (0–4)", 0, 4, 2)

if st.sidebar.button("Predict"):
    X = pd.DataFrame([[
        studytime, failures, absences, G1, G2, Medu, Fedu, traveltime
    ]], columns=["studytime","failures","absences","G1","G2","Medu","Fedu","traveltime"])
    pred = model.predict(X)[0] * 5  # scale 0-100
    st.success(f"Predicted Final Grade (G3) → {pred:.2f}/100")

# Tabs for Dashboard
tab1, tab2, tab3, tab4 = st.tabs(["📊 Charts", "📝 Dataset", "📈 Scatter Plots", "🧪 Evaluation"])

uploaded = st.file_uploader("Upload cleaned_student-mat.csv", type=["csv"])
if uploaded:
    df = pd.read_csv(uploaded)
    df['G1_100'] = df['G1']*5
    df['G2_100'] = df['G2']*5
    df['G3_100'] = df['G3']*5

    # Tab2: Dataset Preview
    with tab2:
        st.subheader("Dataset Preview")
        st.dataframe(df.head(20))
        st.metric("Average G3", f"{df['G3_100'].mean():.2f}")
        st.metric("Max G3", f"{df['G3_100'].max():.0f}")
        st.metric("Min G3", f"{df['G3_100'].min():.0f}")

    # Tab1: Charts
    with tab1:
        st.subheader("Distribution of Final Grades (G3)")
        fig, ax = plt.subplots()
        sns.histplot(df['G3_100'], kde=True, bins=15, color="skyblue", ax=ax)
        ax.set_xlabel("Final Grade (0-100)")
        st.pyplot(fig)

        st.subheader("Boxplot: Study Time vs G3")
        fig, ax = plt.subplots()
        sns.boxplot(x='studytime', y='G3_100', data=df, palette="pastel", ax=ax)
        ax.set_xlabel("Study Time")
        ax.set_ylabel("Final Grade (0-100)")
        st.pyplot(fig)

        st.subheader("Correlation Heatmap")
        numeric_df = df.select_dtypes(include=['int64','float64'])
        fig, ax = plt.subplots(figsize=(10,6))
        sns.heatmap(numeric_df.corr(), annot=False, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

    # Tab3: Scatter Plots
    with tab3:
        st.subheader("Scatter Plots with Jitter")
        fig, ax = plt.subplots()
        x = df['studytime'] + np.random.uniform(-0.1,0.1,len(df))
        ax.scatter(x, df['G3_100'], alpha=0.7, color="green")
        ax.set_xlabel("Study Time")
        ax.set_ylabel("Final Grade (0-100)")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.scatter(df['failures'], df['G3_100'], alpha=0.7, color="orange")
        ax.set_xlabel("Past Failures")
        ax.set_ylabel("Final Grade (0-100)")
        st.pyplot(fig)

        fig, ax = plt.subplots()
        ax.scatter(df['absences'], df['G3_100'], alpha=0.7, color="red")
        ax.set_xlabel("Absences")
        ax.set_ylabel("Final Grade (0-100)")
        st.pyplot(fig)

    # Tab4: Evaluation Metrics
    with tab4:
        st.subheader("Model Evaluation Metrics")
        y_true = df['G3_100']
        X_eval = df[['studytime','failures','absences','G1','G2','Medu','Fedu','traveltime']]
        y_pred = model.predict(X_eval) * 5
        mae = mean_absolute_error(y_true, y_pred)
        rmse = sqrt(mean_squared_error(y_true, y_pred))
        r2 = r2_score(y_true, y_pred)
        st.metric("MAE", f"{mae:.2f}")
        st.metric("RMSE", f"{rmse:.2f}")
        st.metric("R² Score", f"{r2:.2f}")
else:
    st.info("Please upload the cleaned student-mat.csv file to access the dashboard features .")