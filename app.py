import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(page_title="Hospital Dashboard", layout="wide")

# -------------------------------------------------
# SESSION STATE FOR LOGIN
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -------------------------------------------------
# LOGIN PAGE
# -------------------------------------------------
if not st.session_state.logged_in:

    st.title("🔐 Hospital Dashboard Login")

    USERNAME = "admin"
    PASSWORD = "admin123"

    username = st.text_input("Username")
    password = st.text_input("Password", type="password")

    if st.button("Login"):
        if username == USERNAME and password == PASSWORD:
            st.session_state.logged_in = True
            st.success("Login Successful")
            st.experimental_rerun()
        else:
            st.error("Invalid Username or Password")

# -------------------------------------------------
# MAIN DASHBOARD (AFTER LOGIN)
# -------------------------------------------------
else:
    st.title("🏥 Hospital Data Analysis Dashboard")

    # LOGOUT BUTTON
    with st.sidebar:
        st.write("👤 Logged in as: admin")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.experimental_rerun()

    # -------------------------------------------------
    # HOSPITAL SELECTION
    # -------------------------------------------------
    hospital = st.selectbox(
        "Select Hospital",
        ["Hospital A (Patients)", "Hospital B (Appointments)"]
    )

    # -------------------------------------------------
    # LOAD DATASET
    # -------------------------------------------------
    if hospital == "Hospital A (Patients)":
        df = pd.read_csv("patients.csv")
    else:
        df = pd.read_csv("appointments.csv")

    # -------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    # Add Gender if missing
    if "Gender" not in df.columns:
        np.random.seed(42)
        df["Gender"] = np.random.choice(["Male", "Female"], size=len(df))

    # -------------------------------------------------
    # DATA CLEANING
    # -------------------------------------------------
    df.drop_duplicates(inplace=True)
    df.fillna(method="ffill", inplace=True)

    if "age" in df.columns:
        df = df[(df["age"] > 0) & (df["age"] < 120)]

    # -------------------------------------------------
    # SIDEBAR INFO
    # -------------------------------------------------
    st.sidebar.markdown("### 📊 Dashboard Info")
    st.sidebar.write("Selected Hospital:", hospital)
    st.sidebar.write("Total Records:", len(df))

    # -------------------------------------------------
    # SHOW DATA
    # -------------------------------------------------
    st.subheader("📄 Cleaned Dataset Preview")
    st.dataframe(df.head())

    # -------------------------------------------------
    # DASHBOARD FOR HOSPITAL A (PATIENTS)
    # -------------------------------------------------
    if hospital == "Hospital A (Patients)":

        st.header("📊 Patient Analysis Dashboard")

        # Length of stay
        if "arrival_date" in df.columns and "departure_date" in df.columns:
            df["Length_of_Stay"] = (
                df["departure_date"] - df["arrival_date"]
            ).dt.days

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Gender Distribution")
            fig, ax = plt.subplots()
            df["Gender"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

        with col2:
            st.subheader("Age Distribution")
            fig, ax = plt.subplots()
            sns.histplot(df["age"], kde=True, ax=ax)
            st.pyplot(fig)

        st.subheader("Service-wise Patients")
        fig, ax = plt.subplots()
        df["service"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

        if "Length_of_Stay" in df.columns:
            st.subheader("Average Length of Stay by Service")
            fig, ax = plt.subplots()
            df.groupby("service")["Length_of_Stay"].mean().plot(kind="bar", ax=ax)
            st.pyplot(fig)

    # -------------------------------------------------
    # DASHBOARD FOR HOSPITAL B (APPOINTMENTS)
    # -------------------------------------------------
    else:
        st.header("📊 Appointment Analysis Dashboard")

        col1, col2 = st.columns(2)

        with col1:
            st.subheader("Appointment Status")
            fig, ax = plt.subplots()
            df["status"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

        with col2:
            st.subheader("Department-wise Appointments")
            fig, ax = plt.subplots()
            df["department"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

        st.subheader("Appointments by Gender")
        fig, ax = plt.subplots()
        df["Gender"].value_counts().plot(kind="bar", ax=ax)
        st.pyplot(fig)

    # -------------------------------------------------
    # FOOTER
    # -------------------------------------------------
    st.markdown("---")
    st.markdown(
        "**Major Project:** Hospital Data Analysis Dashboard  \n"
        "**Technologies:** Python, Pandas, Matplotlib, Seaborn, Streamlit"
    )
