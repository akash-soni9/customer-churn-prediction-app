import pandas as pd
import numpy as np
import tensorflow as tf
import streamlit as st

from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# TRAIN MODEL (CACHED - RUNS ONLY ONCE)
@st.cache_resource
def train_model():

    # Load dataset
    df = pd.read_csv("Churn_Modelling.csv")

    # Features and target
    X = df.iloc[:, 3:-1].values
    y = df.iloc[:, -1].values

    # Encode Gender
    le = LabelEncoder()
    X[:, 2] = le.fit_transform(X[:, 2])

    # One-Hot Encode Geography (fixed order)
    ct = ColumnTransformer(
        transformers=[
            ('geo', OneHotEncoder(categories=[['France', 'Germany', 'Spain']]), [1])
        ],
        remainder='passthrough'
    )
    X = ct.fit_transform(X)

    # Feature Scaling
    sc = StandardScaler()
    X = sc.fit_transform(X)

    # Train-Test Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=0
    )

    # Build ANN
    ann = tf.keras.models.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    ann.compile(
        optimizer='adam',
        loss='binary_crossentropy',
        metrics=['accuracy']
    )

    # Train ANN
    ann.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        verbose=0
    )

    return ann, le, ct, sc, X_test, y_test



# LOAD TRAINED OBJECTS
ann, le, ct, sc, X_test, y_test = train_model()



# STREAMLIT UI
st.set_page_config(page_title="Churn Prediction App", layout="centered")
st.title("Customer Churn Prediction App")
st.write("Predict whether a customer is likely to leave the bank.")

st.sidebar.header("Enter Customer Details")

credit_score = st.sidebar.number_input("Credit Score", min_value=0)
geography = st.sidebar.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.sidebar.selectbox("Gender", ["Male", "Female"])
age = st.sidebar.number_input("Age", min_value=0)
tenure = st.sidebar.number_input("Tenure", min_value=0)
balance = st.sidebar.number_input("Balance", min_value=0.0)
num_products = st.sidebar.number_input("Number of Products", min_value=1, max_value=4)
has_credit_card = st.sidebar.selectbox("Has Credit Card", [0, 1])
is_active = st.sidebar.selectbox("Is Active Member", [0, 1])
salary = st.sidebar.number_input("Estimated Salary", min_value=0.0)


if st.button("Predict Churn"):

    # Prepare input
    user_data = np.array([[
        credit_score, geography, gender, age, tenure,
        balance, num_products, has_credit_card, is_active, salary
    ]])

    # Apply same preprocessing
    user_data[:, 2] = le.transform(user_data[:, 2])
    user_data = ct.transform(user_data)
    user_data = sc.transform(user_data)

    # Predict probability
    prob = ann.predict(user_data)[0][0]

    # Result
    result = "Churn" if prob > 0.5 else "No Churn"

    st.subheader("Prediction Result")
    st.success(f"Prediction: **{result}**")
    st.info(f"Churn Probability: **{prob:.2f}**")


  
    # MODEL EVALUATION
    st.subheader("Model Performance")

    y_pred = (ann.predict(X_test) > 0.5).astype(int)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)

    st.write(f"**Accuracy:** {accuracy:.2f}")
    st.write("**Confusion Matrix:**")
    st.write(cm)
