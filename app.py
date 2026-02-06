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
    page_title="Hospital Executive Analytics",
    page_icon="🏥",
    layout="wide"
)

sns.set_style("darkgrid")

# ===================== PREMIUM UI =====================
st.markdown("""
<style>

/* Top spacing */
.block-container{
    padding-top:2rem;
}

/* KPI Cards */
div[data-testid="metric-container"]{
    background: linear-gradient(135deg,#0f172a,#020617);
    border: 1px solid #1e293b;
    padding:20px;
    border-radius:14px;
    box-shadow:0 6px 18px rgba(0,0,0,0.5);
    transition:0.3s;
}

div[data-testid="metric-container"]:hover{
    transform:translateY(-6px);
    box-shadow:0 12px 28px rgba(0,0,0,0.7);
}

div[data-testid="metric-container"] *{
    color:white !important;
}

/* Sidebar */
section[data-testid="stSidebar"]{
    background: linear-gradient(180deg,#020617,#0f172a);
    border-right:1px solid #1e293b;
}

/* Buttons */
.stButton>button{
    background-color:#2563eb;
    color:white;
    border-radius:10px;
    border:none;
}

.stButton>button:hover{
    background-color:#1d4ed8;
}

/* Footer */
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

# ===================== SESSION STATE =====================
if "login" not in st.session_state:
    st.session_state.login = False
if "hospital" not in st.session_state:
    st.session_state.hospital = None

# ===================== LOGIN =====================
def login_page():

    st.markdown("<h1 style='text-align:center'>🏥 Hospital Executive Analytics</h1>", unsafe_allow_html=True)
    st.caption("Secure access to hospital insights")

    # IMPORTANT → keys use karo
    with st.form("login_form"):

        user = st.text_input("Username", key="user")
        pwd = st.text_input("Password", type="password", key="pwd")
        hospital = st.selectbox("Select Hospital", ["Hospital1", "Hospital2"], key="hospital_select")

        btn = st.form_submit_button("Login")

    if btn:

        if user == "admin" and pwd == "admin123":

            st.session_state["login"] = True
            st.session_state["hospital"] = hospital

            st.rerun()

        else:
            st.error("Invalid credentials")
     
# ===================== LOAD DATA =====================
if st.session_state.hospital == "Hospital1":
    df = pd.read_csv("patients_final.csv")
else:
    df = pd.read_csv("appointments_final.csv")

# ===================== DATABASE SAFE LOAD =====================
def load_from_db():
    try:
        conn = sqlite3.connect("hospital.db")

        tables = pd.read_sql(
            "SELECT name FROM sqlite_master WHERE type='table';",
            conn
        )

        if tables.empty:
            st.warning("No tables found in database.")
            return pd.DataFrame()

        table_name = tables.iloc[0,0]

        data = pd.read_sql(f"SELECT * FROM {table_name}", conn)
        conn.close()

        return data

    except Exception as e:
        st.error(f"Database Error: {e}")
        return pd.DataFrame()

# ===================== SIDEBAR =====================
st.sidebar.title("📌 Navigation")

page = st.sidebar.radio(
    "Go To",
    ["Dashboard", "Analytics", "Forecasting", "Database"]
)

st.sidebar.write("🏥 Hospital:", st.session_state.hospital)

if st.sidebar.button("Logout"):
    st.session_state.login = False
    st.rerun()

# ===================== KPIs =====================
def patient_kpis(df):
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("👨‍⚕️ Total Patients", len(df))
    c2.metric("🎂 Avg Age", round(df["age"].mean(),1))
    c3.metric("🛏 Beds Available", df["bed_availability"].sum())
    c4.metric("👨 Male Patients", (df["gender"]=="Male").sum())

def appointment_kpis(df):
    c1, c2, c3, c4 = st.columns(4)

    c1.metric("📅 Total Appointments", len(df))
    c2.metric("✅ Completed", (df["status"]=="Completed").sum())
    c3.metric("⏳ Pending", (df["status"]=="Pending").sum())
    c4.metric("🎂 Avg Age", round(df["age"].mean(),1))

# ===================== DASHBOARD =====================
if page == "Dashboard":

    st.markdown("# 🏥 Hospital Executive Dashboard")
    st.caption("Real-time hospital insights and performance metrics")

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
            st.pyplot(fig)

    else:

        appointment_kpis(df)

        col1, col2 = st.columns(2)

        with col1:
            fig, ax = plt.subplots()
            df["department"].value_counts().plot.bar(ax=ax)
            st.pyplot(fig)

        with col2:
            fig, ax = plt.subplots()
            df["status"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            st.pyplot(fig)

# ===================== ANALYTICS =====================
elif page == "Analytics":

    st.title("📊 Hospital Analytics Center")

    tab1, tab2, tab3 = st.tabs(["EDA", "Visualizations", "Correlation"])

    # ---- EDA ----
    with tab1:

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Dataset Preview")
            st.dataframe(df.head())

        with col2:
            st.subheader("Statistical Summary")
            st.dataframe(df.describe())

        st.subheader("Missing Values")
        st.dataframe(df.isnull().sum())

    # ---- Visualizations ----
    with tab2:

        col1, col2 = st.columns(2)

        if "gender" in df.columns:
            fig, ax = plt.subplots()
            df["gender"].value_counts().plot.bar(ax=ax)
            ax.set_title("Gender Distribution")
            col1.pyplot(fig)

        if "department" in df.columns:
            fig, ax = plt.subplots()
            df["department"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
            ax.set_ylabel("")
            col2.pyplot(fig)

        col3, col4 = st.columns(2)

        if "length_of_stay" in df.columns:
            fig, ax = plt.subplots()
            df["length_of_stay"].plot(ax=ax)
            ax.set_title("Length of Stay Trend")
            col3.pyplot(fig)

        if "service" in df.columns:
            fig, ax = plt.subplots()
            df["service"].value_counts().plot.bar(ax=ax)
            ax.set_title("Service Usage")
            col4.pyplot(fig)

    # ---- Correlation ----
    with tab3:

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
            st.warning("Required columns not available.")

# ===================== FORECASTING =====================
elif page == "Forecasting":

    st.title("🔮 Forecasting")

    num_cols = df.select_dtypes(include=np.number).columns

    target = st.selectbox("Select Target Column", num_cols)

    if df[target].dropna().shape[0] > 20:

        model = ARIMA(df[target].dropna(), order=(1,1,1))
        fit = model.fit()

        forecast = fit.forecast(10)

        st.write("Next 10 Predictions:")
        st.write(forecast)

        fig, ax = plt.subplots()
        ax.plot(df[target], label="Actual")
        ax.plot(range(len(df), len(df)+10), forecast, label="Forecast")
        ax.legend()
        st.pyplot(fig)

    else:
        st.warning("Not enough data for forecasting.")

# ===================== DATABASE =====================
elif page == "Database":

    st.title("🗄 SQLite Database View")

    db_df = load_from_db()

    if not db_df.empty:
        st.dataframe(db_df)

# ===================== FOOTER =====================
st.markdown(f"""
<div class="footer">
Hospital Executive Analytics v3.0 | Hospital: <b>{st.session_state.hospital}</b> | © 2026 Diksha Tiwari
</div>
""", unsafe_allow_html=True)

