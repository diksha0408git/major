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

# ===================== GLOBAL UI =====================
st.markdown("""
<style>

.main {
    background-color:#020617;
}

.block-container{
    padding-top:2rem;
}

.stMetric{
    background-color:#0f172a;
    padding:18px;
    border-radius:12px;
    text-align:center;
    box-shadow:0 4px 12px rgba(0,0,0,0.4);
}

.footer {
    position: fixed;
    left: 0;
    bottom: 0;
    width: 100%;
    background-color: #020617;
    color: #94a3b8;
    text-align: center;
    padding: 10px;
    font-size: 13px;
    border-top: 1px solid #1e293b;
}

</style>
""", unsafe_allow_html=True)

sns.set_style("darkgrid")

# ===================== SESSION STATE =====================
if "login" not in st.session_state:
    st.session_state.login = False
if "hospital" not in st.session_state:
    st.session_state.hospital = None

# ===================== LOGIN PAGE =====================
def login_page():

    st.markdown("""
    <style>
    .login-box {
        background-color:#0f172a;
        padding:35px;
        border-radius:14px;
        width:420px;
        margin:auto;
        box-shadow:0 0 30px rgba(0,0,0,0.6);
    }
    </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:white;text-align:center'>🔐 Hospital Login</h2>", unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        hospital = st.selectbox("Select Hospital", ["Hospital1", "Hospital2"])
        btn = st.form_submit_button("Login")

    if btn:
        if username == "admin" and password == "admin123":
            st.session_state.login = True
            st.session_state.hospital = hospital
            st.rerun()
        else:
            st.error("Invalid Credentials")

    st.markdown("</div>", unsafe_allow_html=True)

if not st.session_state.login:
    login_page()
    st.stop()

# ===================== LOAD DATA =====================
if st.session_state.hospital == "Hospital1":
    df = pd.read_csv("patients_final.csv")
else:
    df = pd.read_csv("appointments_final.csv")

# ===================== SQLITE CONNECTION =====================
def load_from_db():
    conn = sqlite3.connect("hospital.db")
    data = pd.read_sql("SELECT * FROM hospital_data", conn)
    conn.close()
    return data

# ===================== SIDEBAR =====================
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go To",
    ["Dashboard", "Analytics", "Forecasting", "Database"]
)

st.sidebar.markdown("---")
st.sidebar.write("🏥 Hospital:", st.session_state.hospital)

if st.sidebar.button("Logout"):
    st.session_state.login = False
    st.rerun()

# ===================== KPI FUNCTIONS =====================
def patient_kpis(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Patients", len(df))
    c2.metric("Average Age", round(df["age"].mean(), 1))
    c3.metric("Male Patients", (df["gender"] == "Male").sum())
    c4.metric("Beds Available", df["bed_availability"].sum())

def appointment_kpis(df):
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Appointments", len(df))
    c2.metric("Completed", (df["status"] == "Completed").sum())
    c3.metric("Pending", (df["status"] == "Pending").sum())
    c4.metric("Avg Age", round(df["age"].mean(), 1))

# ===================== DASHBOARD =====================
if page == "Dashboard":

    st.title("🏥 Hospital Executive Dashboard")

    if st.session_state.hospital == "Hospital1":
        patient_kpis(df)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            ax.hist(df["age"], bins=20)
            ax.set_title("Age Distribution")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            df["gender"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title("Gender Split")
            st.pyplot(fig)

    else:
        appointment_kpis(df)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            df["department"].value_counts().plot.bar(ax=ax)
            ax.set_title("Appointments by Department")
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            df["status"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            ax.set_title("Appointment Status")
            st.pyplot(fig)

# ===================== ANALYTICS CENTER =====================
elif page == "Analytics":

    st.title("📊 Hospital Analytics Center")

    tab1, tab2, tab3 = st.tabs(["EDA", "Visualizations", "Correlation"])

    # ----------- EDA -----------
    with tab1:

        c1, c2 = st.columns(2)

        with c1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head())

        with c2:
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())

        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum().to_frame("Missing Count"))

    # ----------- VISUALIZATIONS -----------
    with tab2:

        col1, col2 = st.columns(2)

        if "gender" in df.columns:
            with col1:
                fig, ax = plt.subplots()
                df["gender"].value_counts().plot.bar(ax=ax)
                ax.set_title("Gender Distribution")
                st.pyplot(fig)

        if "department" in df.columns:
            with col2:
                fig, ax = plt.subplots()
                df["department"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
                ax.set_ylabel("")
                ax.set_title("Department Share")
                st.pyplot(fig)

        col3, col4 = st.columns(2)

        if "length_of_stay" in df.columns:
            with col3:
                fig, ax = plt.subplots()
                df["length_of_stay"].plot(ax=ax)
                ax.set_title("Length of Stay Trend")
                ax.set_xlabel("Records")
                ax.set_ylabel("Days")
                st.pyplot(fig)

        if "service" in df.columns:
            with col4:
                fig, ax = plt.subplots()
                df["service"].value_counts().plot.bar(ax=ax)
                ax.set_title("Service Usage")
                st.pyplot(fig)

    # ----------- CORRELATION -----------
    with tab3:

        st.subheader("Correlation Matrix")

        if "length_of_stay" in df.columns and "age" in df.columns:

            fig, ax = plt.subplots(figsize=(8,5))
            sns.heatmap(
                df[["length_of_stay","age"]].corr(),
                annot=True,
                cmap="coolwarm",
                ax=ax
            )
            st.pyplot(fig)

        else:
            st.warning("Required columns not found for correlation.")

# ===================== FORECASTING =====================
elif page == "Forecasting":

    st.title("🔮 Forecasting (ARIMA)")

    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    if len(num_cols) < 1:
        st.warning("No numeric column available for forecasting")
    else:
        target = st.selectbox("Select Target Column", num_cols)

        if df[target].dropna().shape[0] < 20:
            st.warning("Not enough data for forecasting")
        else:
            model = ARIMA(df[target].dropna(), order=(1,1,1))
            model_fit = model.fit()
            forecast = model_fit.forecast(steps=10)

            st.subheader("Next 10 Predictions")
            st.write(forecast)

            fig, ax = plt.subplots()
            ax.plot(df[target], label="Actual")
            ax.plot(range(len(df), len(df)+10), forecast, label="Forecast")
            ax.legend()
            st.pyplot(fig)

# ===================== DATABASE =====================
elif page == "Database":

    st.title("🗄 SQLite Database View")
    db_df = load_from_db()
    st.dataframe(db_df)

# ===================== FOOTER =====================
st.markdown(f"""
<div class="footer">
🏥 Hospital Analytics Dashboard v2.0 |
Hospital: <b>{st.session_state.hospital}</b> |
Logged in as: <b>Admin</b> |
© 2026 Diksha Tiwari
</div>
""", unsafe_allow_html=True)
