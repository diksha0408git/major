import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

# -------------------- PAGE CONFIG --------------------
st.set_page_config(
    page_title="Hospital Analytics Dashboard",
    page_icon="🏥",
    layout="wide"
)

# -------------------- SESSION --------------------
if "login" not in st.session_state:
    st.session_state.login = False
if "hospital" not in st.session_state:
    st.session_state.hospital = None

# -------------------- LOGIN UI --------------------
if not st.session_state.login:
    st.markdown("## 🔐 Hospital Login")

    col1, col2 = st.columns(2)
    with col1:
        username = st.text_input("Username")
        hospital = st.selectbox("Select Hospital", ["Hospital1", "Hospital2"])
    with col2:
        password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == "admin" and password == "admin123":
            st.session_state.login = True
            st.session_state.hospital = hospital
            st.rerun()
        else:
            st.error("Invalid credentials")

    st.stop()

# -------------------- SIDEBAR --------------------
st.sidebar.title("🏥 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["Dashboard", "EDA", "Forecasting", "Database View"]
)

# -------------------- LOAD DATA --------------------
@st.cache_data
def load_data():
    patients = pd.read_csv("patients_final.csv")
    appointments = pd.read_csv("appointments_final.csv")
    return patients, appointments

patients_df, appointments_df = load_data()

# -------------------- SQLITE --------------------
conn = sqlite3.connect("hospital.db", check_same_thread=False)
cursor = conn.cursor()

def create_tables():
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS patients (
        patient_id INTEGER,
        name TEXT,
        age INTEGER,
        gender TEXT,
        department TEXT,
        bed_availability INTEGER
    )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS appointments (
        appointment_id INTEGER,
        patient_id INTEGER,
        appointment_date TEXT,
        department TEXT,
        gender TEXT,
        discharge_date TEXT
    )
    """)
    conn.commit()

create_tables()

patients_df.to_sql("patients", conn, if_exists="replace", index=False)
appointments_df.to_sql("appointments", conn, if_exists="replace", index=False)

def load_from_db(table):
    return pd.read_sql(f"SELECT * FROM {table}", conn)

# -------------------- DASHBOARD --------------------
if page == "Dashboard":
    st.title(f"📊 {st.session_state.hospital} Dashboard")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Patients", patients_df.shape[0])
    col2.metric("Total Appointments", appointments_df.shape[0])
    col3.metric("Departments", patients_df["department"].nunique())

    st.subheader("Patients by Department")
    fig, ax = plt.subplots()
    patients_df["department"].value_counts().plot(kind="bar", ax=ax)
    st.pyplot(fig)

    st.subheader("Gender Distribution")
    fig, ax = plt.subplots()
    patients_df["gender"].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
    ax.set_ylabel("")
    st.pyplot(fig)

# -------------------- EDA --------------------
elif page == "EDA":
    st.title("📈 Exploratory Data Analysis")

    st.subheader("Age Distribution")
    fig, ax = plt.subplots()
    sns.histplot(patients_df["age"], kde=True, ax=ax)
    st.pyplot(fig)

    st.subheader("Age vs Bed Availability")
    fig, ax = plt.subplots()
    sns.scatterplot(
        x="age",
        y="bed_availability",
        data=patients_df,
        ax=ax
    )
    st.pyplot(fig)

    st.subheader("Box Plot (Age)")
    fig, ax = plt.subplots()
    sns.boxplot(y=patients_df["age"], ax=ax)
    st.pyplot(fig)

    st.subheader("Correlation Heatmap")
    fig, ax = plt.subplots()
    sns.heatmap(
        patients_df.select_dtypes(include=np.number).corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

# -------------------- FORECASTING --------------------
elif page == "Forecasting":
    st.title("🔮 Appointment Forecasting")

    appointments_df["appointment_date"] = pd.to_datetime(
        appointments_df["appointment_date"]
    )

    daily = appointments_df.groupby(
        appointments_df["appointment_date"].dt.date
    ).size()

    st.subheader("Historical Appointments")
    fig, ax = plt.subplots()
    daily.plot(ax=ax)
    st.pyplot(fig)

    # Simple Moving Average Forecast
    forecast_days = 7
    forecast = daily.rolling(3).mean().iloc[-1]

    st.success(f"📌 Expected average appointments per day: {int(forecast)}")

# -------------------- DATABASE VIEW --------------------
elif page == "Database View":
    st.title("🗄 SQLite Database Tables")

    st.subheader("Patients Table")
    st.dataframe(load_from_db("patients"))

    st.subheader("Appointments Table")
    st.dataframe(load_from_db("appointments"))

# -------------------- LOGOUT --------------------
st.sidebar.button(
    "Logout",
    on_click=lambda: st.session_state.update({"login": False})
)

# -------------------- FOOTER --------------------
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
    🏥 Hospital Analytics Dashboard | Hospital: <b>{st.session_state.hospital}</b> |
    User: <b>Admin</b> | © 2026 Diksha Tiwari
</div>
""", unsafe_allow_html=True)

