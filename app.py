# ===================== IMPORTS =====================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
from statsmodels.tsa.arima.model import ARIMA

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Hospital Analytics Dashboard",
    page_icon="🏥",
    layout="wide"
)

# ===================== SESSION STATE =====================
if "login" not in st.session_state:
    st.session_state.login = False
if "hospital" not in st.session_state:
    st.session_state.hospital = None

# ===================== LOGIN PAGE =====================
def login():
    st.markdown(
        """
        <style>
        .login-box {
            width: 350px;
            margin: 100px auto;
            padding: 30px;
            border-radius: 10px;
            background-color: #f5f5f5;
            box-shadow: 0px 4px 10px rgba(0,0,0,0.1);
        }
        .login-title {
            text-align: center;
            font-size: 26px;
            font-weight: bold;
            margin-bottom: 20px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown('<div class="login-box">', unsafe_allow_html=True)
    st.markdown('<div class="login-title">Hospital Login</div>', unsafe_allow_html=True)

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    hospital = st.selectbox("Select Hospital", ["Hospital1", "Hospital2"])

    if st.button("Login", use_container_width=True):
        if username == "admin" and password == "1234":
            st.session_state["logged_in"] = True
            st.session_state["hospital"] = hospital
            st.rerun()
        else:
            st.error("Invalid username or password")

    st.markdown('</div>', unsafe_allow_html=True)


# session state
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False
if "hospital" not in st.session_state:
    st.session_state["hospital"] = None

# show login if not logged in
if not st.session_state["logged_in"]:
    login()
    st.stop()

# ===================== LOAD DATA =====================
if st.session_state.hospital == "Hospital1":
    df = pd.read_csv("appointments_final.csv")   # hospital1 → appointments
    table_name = "appointments"
else:
    df = pd.read_csv("patients_final.csv")       # hospital2 → patients
    table_name = "patients"

# ===================== SQLITE =====================
conn = sqlite3.connect("hospital_major_project.db", check_same_thread=False)
df.to_sql(table_name, conn, if_exists="replace", index=False)

def load_from_db(table):
    return pd.read_sql(f"SELECT * FROM {table}", conn)

# ===================== SIDEBAR =====================
st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "EDA", "Visualizations", "Correlation", "Forecasting", "Database"]
)

st.sidebar.markdown("---")
st.sidebar.write("Hospital:", st.session_state["hospital"])

# Logout button
if st.sidebar.button("Logout"):
    st.session_state.clear()   # clears all session data
    st.rerun()

# ===================== KPI =====================
st.title("Hospital Dashboard")

if st.session_state.hospital == "Hospital1":
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Appointments", len(df))
    c2.metric("Departments", df["department"].nunique())
    c3.metric("Completed", (df["status"] == "Completed").sum())
    c4.metric("Pending", (df["status"] == "Pending").sum())

else:
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", len(df))
    c2.metric("Average Age", round(df["age"].mean(), 1))
    c3.metric("Departments", df["department"].nunique())
    c4.metric("Beds Available", df["bed_availability"].sum())

# ===================== CLEAN COLUMNS FOR CHARTS =====================
ignore_cols = [c for c in df.columns if "id" in c.lower() or "date" in c.lower()]
num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in ignore_cols]
cat_cols = [c for c in df.select_dtypes(include="object").columns if c not in ignore_cols]

# ===================== DASHBOARD =====================
if page == "Dashboard":

    col1, col2 = st.columns(2)

    if len(cat_cols) > 0:
        with col1:
            fig, ax = plt.subplots()
            df[cat_cols[0]].value_counts().plot.bar(ax=ax)
            ax.set_title("Bar Chart")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            df[cat_cols[0]].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title("Pie Chart")
            st.pyplot(fig)

# ===================== EDA =====================
elif page == "EDA":
    st.title("Exploratory Data Analysis")
    clean_df = df.dropna()
    st.dataframe(clean_df.head())
    st.dataframe(clean_df.describe())

# ===================== VISUALIZATIONS =====================
elif page == "Visualizations":
    st.title("Advanced Visualizations")

    chart = st.selectbox(
        "Select Chart Type",
        ["Histogram", "Bar Chart", "Pie Chart", "Scatter Plot", "Box Plot"]
    )

    if chart == "Histogram":
        col = st.selectbox("Column", num_cols)
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=20)
        st.pyplot(fig)

    elif chart == "Bar Chart":
        col = st.selectbox("Column", cat_cols)
        fig, ax = plt.subplots()
        df[col].value_counts().plot.bar(ax=ax)
        st.pyplot(fig)

    elif chart == "Pie Chart":
        col = st.selectbox("Column", cat_cols)
        fig, ax = plt.subplots()
        df[col].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    elif chart == "Scatter Plot":
        x = st.selectbox("X Axis", num_cols)
        y = st.selectbox("Y Axis", num_cols)
        fig, ax = plt.subplots()
        ax.scatter(df[x], df[y])
        st.pyplot(fig)

    elif chart == "Box Plot":
        col = st.selectbox("Column", num_cols)
        fig, ax = plt.subplots()
        ax.boxplot(df[col])
        st.pyplot(fig)

# ===================== CORRELATION =====================
elif page == "Correlation":
    st.title("Correlation Heatmap")
    num_df = df.select_dtypes(include=np.number)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.heatmap(num_df.corr(), annot=True, cmap="RdYlBu", ax=ax)
    st.pyplot(fig)

# ===================== FORECASTING =====================
elif page == "Forecasting":
    st.title("Forecasting (ARIMA)")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    target = st.selectbox("Select Column", num_cols)

    series = df[target].dropna()

    if len(series) < 20:
        st.warning("Not enough data for forecasting")
    else:
        model = ARIMA(series, order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)

        st.write("Next 10 Predictions")
        st.write(forecast)

        fig, ax = plt.subplots()
        ax.plot(series.values, label="Actual")
        ax.plot(range(len(series), len(series)+10), forecast, label="Forecast")
        ax.legend()
        st.pyplot(fig)

# ===================== DATABASE =====================
elif page == "Database":
    st.title("🗄 Database View")
    st.dataframe(load_from_db(table_name))

# ===================== FOOTER =====================
st.markdown("<br><br><br>", unsafe_allow_html=True)
st.markdown(f"""
<style>
.footer {{
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #020617;
    color: #94a3b8;
    text-align: center;
    padding: 10px;
    font-size: 13px;
}}
</style>

<div class="footer">
     Hospital Analytics Dashboard |
    Hospital: <b>{st.session_state.hospital}</b> |
    © 2026 Diksha Tiwari
</div>
""", unsafe_allow_html=True)


