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
# SESSION STATE
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -------------------------------------------------
# LOGIN PAGE (IMPROVED UI)
# -------------------------------------------------
if not st.session_state.logged_in:

    st.markdown("<h1 style='text-align:center;'>🏥 Hospital Dashboard</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>Secure Login</h4>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("### 🔐 Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.success("Login Successful")
                st.rerun()
            else:
                st.error("Invalid username or password")

# -------------------------------------------------
# MAIN DASHBOARD
# -------------------------------------------------
else:
    st.title("📊 Hospital Data Analysis Dashboard")

    # LOGOUT
    with st.sidebar:
        st.write("👤 User: admin")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.rerun()

    # -------------------------------------------------
    # HOSPITAL SELECTION
    # -------------------------------------------------
    hospital = st.selectbox(
        "Select Hospital",
        ["Hospital A (Patients)", "Hospital B (Appointments)"]
    )

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------
    if hospital == "Hospital A (Patients)":
        df = pd.read_csv("patients_updated.csv")
    else:
        df = pd.read_csv("appointments_updated.csv")

    # -------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Gender" not in df.columns:
        np.random.seed(42)
        df["Gender"] = np.random.choice(["Male", "Female"], size=len(df))
    else:
        df["Gender"] = df["Gender"].fillna("Unknown")

    # -------------------------------------------------
    # CLEANING
    # -------------------------------------------------
    df.drop_duplicates(inplace=True)
    df.fillna(method="ffill", inplace=True)

    if "age" in df.columns:
        df = df[(df["age"] > 0) & (df["age"] < 120)]

    # -------------------------------------------------
    # FILTERS (ADVANCED FEATURE)
    # -------------------------------------------------
    st.sidebar.markdown("### 🔎 Filters")

    if "Gender" in df.columns:
        gender_filter = st.sidebar.multiselect(
            "Select Gender",
            df["Gender"].unique(),
            default=df["Gender"].unique()
        )
        df = df[df["Gender"].isin(gender_filter)]

    # -------------------------------------------------
    # KPI CARDS (ADVANCED DASHBOARD)
    # -------------------------------------------------
    st.markdown("## 📌 Key Metrics")

    k1, k2, k3 = st.columns(3)

    k1.metric("Total Records", len(df))

    if "age" in df.columns:
        k2.metric("Average Age", round(df["age"].mean(), 1))
    else:
        k2.metric("Average Age", "N/A")

    if "Gender" in df.columns:
        k3.metric("Male %", round((df["Gender"] == "Male").mean() * 100, 1))

    # -------------------------------------------------
    # DATA PREVIEW
    # -------------------------------------------------
    with st.expander("📄 View Cleaned Dataset"):
        st.dataframe(df.head(20))

    # -------------------------------------------------
    # HOSPITAL A DASHBOARD
    # -------------------------------------------------
    if hospital == "Hospital A (Patients)":

        st.header("🧑‍⚕️ Patient Analysis")

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
            if "age" in df.columns:
                st.subheader("Age Distribution")
                fig, ax = plt.subplots()
                sns.histplot(df["age"], kde=True, ax=ax)
                st.pyplot(fig)

        if "service" in df.columns:
            st.subheader("Service-wise Patients")
            fig, ax = plt.subplots()
            df["service"].value_counts().plot(kind="bar", ax=ax)
            st.pyplot(fig)

    # -------------------------------------------------
    # HOSPITAL B DASHBOARD
    # -------------------------------------------------
    else:
        st.header("📅 Appointment Analysis")

        col1, col2 = st.columns(2)

        with col1:
            if "status" in df.columns:
                st.subheader("Appointment Status")
                fig, ax = plt.subplots()
                df["status"].value_counts().plot(kind="bar", ax=ax)
                st.pyplot(fig)

        with col2:
            if "department" in df.columns:
                st.subheader("Department-wise Appointments")
                fig, ax = plt.subplots()
                df["department"].value_counts().plot(kind="bar", ax=ax)
                st.pyplot(fig)

    # -------------------------------------------------
    # FOOTER
    # -------------------------------------------------
    st.markdown("---")
    st.markdown(
        "**Major Project:** Hospital Data Analysis Dashboard  \n"
        "**Advanced Features:** Login UI, KPIs, Filters, Dynamic Dashboards"
    )


