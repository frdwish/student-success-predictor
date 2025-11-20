#importing all the modules

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
import joblib

# Create folders if they don't exist
Path("data").mkdir(exist_ok=True)
Path("model").mkdir(exist_ok=True)


# Load Raw Dataset
df = pd.read_csv("data/raw_student-mat.csv", sep=";")
df.head()


#Basic Data Info
df.info()
df.describe()
df.isnull().sum()
df.duplicated().sum()

#  Identify Columns /Separate categorical and numerical columns for EDA. 
cat_cols = df.select_dtypes(include="object").columns.tolist()
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
num_cols

#Showing Categorical value counts
for col in cat_cols:
    print(f"{col} ({df[col].nunique()} unique)\n{df[col].value_counts()}\n")

#Plot Numerical Distributions
df[num_cols].hist(bins=20, figsize=(14,10))
plt.show()

#Final Grade Distribution/ Visualize G3 (final grade) distribution with histogram + KDE plot.
sns.histplot(df["G3"], kde=True, bins=15)
plt.title("Final Grade Distribution (G3)")
plt.show()

# Correlation Heatmap
plt.figure(figsize=(14,10))
sns.heatmap(df[num_cols].corr(), annot=True, cmap="coolwarm")
plt.show()

#Data Cleaning
binary_map = {"yes":1, "no":0}
binary_cols = ["schoolsup","famsup","paid","activities","nursery","higher","internet","romantic"]

for col in binary_cols:
    if col in df.columns:
        df[col] = df[col].map(binary_map)

#IQR Outlier Removal (absences)
Q1 = df["absences"].quantile(0.25)
Q3 = df["absences"].quantile(0.75)
IQR = Q3 - Q1
low = Q1 - 1.5*IQR
high = Q3 + 1.5*IQR

df_clean = df[(df["absences"] >= low) & (df["absences"] <= high)]
df_clean.shape

#Feature Selection
feature_cols = [
    "studytime","failures","absences","G1","G2","Medu","Fedu","traveltime"
]
target = "G3"

X = df_clean[feature_cols]
y = df_clean[target]

#Train-Test Split
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

#Train RandomForest Model
from sklearn.ensemble import RandomForestRegressor

model = RandomForestRegressor(
    n_estimators=300,
    random_state=42
)

model.fit(X_train, y_train)

#Model Evaluation
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from math import sqrt


y_pred = model.predict(X_test)
rmse = sqrt(mean_squared_error(y_test, y_pred))
print("RMSE:", rmse)
print("MAE :", mean_absolute_error(y_test, y_pred))
print("R2  :", r2_score(y_test, y_pred))

#Save Cleaned Dataset + Model
df_clean.to_csv("data/cleaned_student-mat.csv", index=False)
joblib.dump(model, "model/student_model.pkl")

print("Saved cleaned dataset & model!")
