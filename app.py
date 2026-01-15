import pandas as pd
import streamlit as st
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Title
st.title("🌧️ Rainfall Prediction App")

# Load dataset
df = pd.read_csv("weatherAUS.csv")

# Preprocessing
df['RainTomorrow'] = df['RainTomorrow'].map({'Yes': 1, 'No': 0})
df = df.dropna(subset=['RainTomorrow'])

X = df.select_dtypes(include=['float64'])
X = X.fillna(X.mean())
y = df['RainTomorrow']

# Train model
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# User input
st.header("Enter Weather Values")

user_input = {}
for col in X.columns:
    user_input[col] = st.number_input(col, value=float(X[col].mean()))

input_df = pd.DataFrame([user_input])

# Prediction
if st.button("Predict Rain"):
    prediction = model.predict(input_df)[0]
    if prediction == 1:
        st.success("🌧️ It will rain tomorrow")
    else:
        st.success("☀️ No rain tomorrow")
