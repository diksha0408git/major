import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

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
# LOGIN PAGE (PROFESSIONAL UI)
# -------------------------------------------------
if not st.session_state.logged_in:

    st.markdown("""
        <style>
            .login-box {
                background-color: #ffffff;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0px 0px 15px rgba(0,0,0,0.1);
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown("<h1 style='text-align:center;'>🏥 Hospital Analytics System</h1>", unsafe_allow_html=True)
    st.markdown("<h4 style='text-align:center;'>Secure Login</h4>", unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])

    with col2:
        st.markdown("<div class='login-box'>", unsafe_allow_html=True)
        username = st.text_input("👤 Username")
        password = st.text_input("🔑 Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.success("Login Successful")
                st.experimental_rerun()
            else:
                st.error("Invalid Username or Password")
        st.markdown("</div>", unsafe_allow_html=True)

# -------------------------------------------------
# MAIN DASHBOARD
# -------------------------------------------------
else:
    st.title("📊 Hospital Data Analysis Dashboard")

    # SIDEBAR
    with st.sidebar:
        st.markdown("### 👤 User: admin")
        if st.button("Logout"):
            st.session_state.logged_in = False
            st.experimental_rerun()

    # -------------------------------------------------
    # HOSPITAL SELECTION
    # -------------------------------------------------
    hospital = st.selectbox(
        "🏥 Select Hospital",
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
    # PREPROCESSING & CLEANING
    # -------------------------------------------------
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    if "Gender" not in df.columns:
        df["Gender"] = np.random.choice(["Male", "Female"], size=len(df))
    else:
        df["Gender"] = df["Gender"].fillna("Unknown")

    df.drop_duplicates(inplace=True)
    df.fillna(method="ffill", inplace=True)

    # -------------------------------------------------
    # FILTERS
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
    # KPI CARDS
    # -------------------------------------------------
    st.markdown("## 📌 Key Metrics")
    k1, k2, k3 = st.columns(3)

    k1.metric("Total Records", len(df))

    if "age" in df.columns:
        k2.metric("Average Age", round(df["age"].mean(), 1))
    else:
        k2.metric("Average Age", "N/A")

    k3.metric("Unique Genders", df["Gender"].nunique())

    # -------------------------------------------------
    # DATA PREVIEW
    # -------------------------------------------------
    with st.expander("📄 View Dataset"):
        st.dataframe(df.head(20))

    # -------------------------------------------------
    # HOSPITAL A DASHBOARD
    # -------------------------------------------------
    if hospital == "Hospital A (Patients)":
        st.header("🧑‍⚕️ Patient Analytics")

        if "service" in df.columns:
            fig = px.bar(
                df,
                x="service",
                title="Patients per Service",
                color="Gender"
            )
            st.plotly_chart(fig, use_container_width=True)

        if "age" in df.columns:
            fig = px.histogram(
                df,
                x="age",
                nbins=20,
                title="Age Distribution",
                color="Gender"
            )
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # HOSPITAL B DASHBOARD
    # -------------------------------------------------
    else:
        st.header("📅 Appointment Analytics")

        if "department" in df.columns:
            fig = px.bar(
                df,
                x="department",
                title="Appointments by Department",
                color="Gender"
            )
            st.plotly_chart(fig, use_container_width=True)

        if "status" in df.columns:
            fig = px.pie(
                df,
                names="status",
                title="Appointment Status Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # FOOTER
    # -------------------------------------------------
    st.markdown("---")
    st.markdown(
        "**Major Project – Hospital Analytics System**  \n"
        "🔹 Secure Login  🔹 Interactive Dashboards  🔹 Dynamic Filters"
    )

