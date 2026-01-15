# ===================== IMPORTS =====================
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from statsmodels.tsa.arima.model import ARIMA

# ===================== PAGE CONFIG =====================
st.set_page_config(
    page_title="Hospital Analytics System",
    page_icon="🏥",
    layout="wide"
)

# ===================== LOGIN UI =====================
def login_page():
    st.markdown("""
        <style>
        .login-box {
            background-color:#0f172a;
            padding:30px;
            border-radius:12px;
            width:420px;
            margin:auto;
            box-shadow:0px 0px 25px rgba(0,0,0,0.6);
        }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<div class='login-box'>", unsafe_allow_html=True)
    st.markdown("<h2 style='color:white;text-align:center'>🔐 Hospital Login</h2>", unsafe_allow_html=True)

    with st.form("login_form"):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        hospital = st.text_input("Hospital Name (Hospital1 / Hospital2)")
        login = st.form_submit_button("Login")

    if login:
        if username == "admin" and password == "admin123":
            st.session_state.logged_in = True
            st.session_state.hospital = hospital
            st.success("Login Successful")
            st.rerun()
        else:
            st.error("Invalid Credentials")

    st.markdown("</div>", unsafe_allow_html=True)

if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

if not st.session_state.logged_in:
    login_page()
    st.stop()

# ===================== LOAD DATA =====================
def load_dataset(hospital):
    if hospital.lower() == "hospital1":
        return pd.read_csv("patients_final.csv")
    elif hospital.lower() == "hospital2":
        return pd.read_csv("appointments_final.csv")
    else:
        return None

df = load_dataset(st.session_state.hospital)

if df is None:
    st.error("Invalid Hospital Name")
    st.stop()

# ===================== SIDEBAR =====================
st.sidebar.title("🏥 Dashboard Menu")
st.sidebar.write(f"Hospital: **{st.session_state.hospital}**")

menu = st.sidebar.radio(
    "Select Module",
    ["EDA", "Visualization", "Correlation", "Forecasting"]
)

if st.sidebar.button("Logout"):
    st.session_state.logged_in = False
    st.rerun()

# ===================== EDA =====================
def eda_section(df):
    st.subheader("📊 Exploratory Data Analysis")

    c1, c2, c3 = st.columns(3)
    c1.metric("Rows", df.shape[0])
    c2.metric("Columns", df.shape[1])
    c3.metric("Missing Values", df.isnull().sum().sum())

    st.markdown("### Dataset Preview")
    st.dataframe(df.head())

    st.markdown("### Statistical Summary")
    st.dataframe(df.describe())

    st.markdown("### Missing Values")
    st.write(df.isnull().sum())

# ===================== VISUALIZATION =====================
def visualization_section(df):
    st.subheader("📈 Advanced Visualizations")

    num_cols = df.select_dtypes(include=np.number).columns
    cat_cols = df.select_dtypes(include='object').columns

    chart = st.selectbox(
        "Select Chart Type",
        ["Histogram", "Bar Chart", "Pie Chart", "Scatter Plot", "Box Plot"]
    )

    if chart == "Histogram":
        col = st.selectbox("Column", num_cols)
        fig, ax = plt.subplots()
        ax.hist(df[col], bins=25)
        st.pyplot(fig)

    elif chart == "Bar Chart":
        col = st.selectbox("Column", cat_cols)
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    elif chart == "Pie Chart":
        col = st.selectbox("Column", cat_cols)
        fig, ax = plt.subplots()
        df[col].value_counts().plot(kind="pie", autopct="%1.1f%%", ax=ax)
        ax.set_ylabel("")
        st.pyplot(fig)

    elif chart == "Scatter Plot":
        x = st.selectbox("X-axis", num_cols)
        y = st.selectbox("Y-axis", num_cols)
        fig, ax = plt.subplots()
        ax.scatter(df[x], df[y])
        st.pyplot(fig)

    elif chart == "Box Plot":
        col = st.selectbox("Column", num_cols)
        fig, ax = plt.subplots()
        ax.boxplot(df[col])
        st.pyplot(fig)

# ===================== CORRELATION =====================
def correlation_section(df):
    st.subheader("🔥 Correlation Heatmap")

    num_df = df.select_dtypes(include=np.number)

    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(
        num_df.corr(),
        annot=True,
        cmap="coolwarm",
        ax=ax
    )
    st.pyplot(fig)

# ===================== FORECASTING =====================
def forecasting_section(df):
    st.subheader("🔮 Forecasting")

    num_cols = df.select_dtypes(include=np.number).columns
    target = st.selectbox("Target Column", num_cols)

    try:
        model = ARIMA(df[target], order=(1,1,1))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)

        st.markdown("### Future 10 Predictions")
        st.write(forecast)

        fig, ax = plt.subplots()
        ax.plot(df[target], label="Actual")
        ax.plot(range(len(df), len(df)+10), forecast, label="Forecast")
        ax.legend()
        st.pyplot(fig)

    except:
        st.error("Forecasting not possible for this column")

# ===================== MAIN =====================
st.title("🏥 Hospital Analytics Dashboard")

if menu == "EDA":
    eda_section(df)

elif menu == "Visualization":
    visualization_section(df)

elif menu == "Correlation":
    correlation_section(df)

elif menu == "Forecasting":
    forecasting_section(df)

