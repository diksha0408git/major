import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

# -------------------------------------------------
# PAGE CONFIG
# -------------------------------------------------
st.set_page_config(
    page_title="Hospital Analytics System",
    layout="wide"
)

# -------------------------------------------------
# SESSION STATE
# -------------------------------------------------
if "logged_in" not in st.session_state:
    st.session_state.logged_in = False

# -------------------------------------------------
# LOGIN PAGE
# -------------------------------------------------
if not st.session_state.logged_in:

    st.markdown("""
        <style>
        .login-box {
            background-color: #ffffff;
            padding: 35px;
            border-radius: 12px;
            box-shadow: 0px 0px 20px rgba(0,0,0,0.15);
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
        password = st.text_input("🔐 Password", type="password")

        if st.button("Login"):
            if username == "admin" and password == "admin123":
                st.session_state.logged_in = True
                st.success("Login Successful")
                st.rerun()
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
            st.rerun()

    # -------------------------------------------------
    # HOSPITAL SELECTION
    # -------------------------------------------------
    hospital = st.selectbox(
        "🏥 Select Hospital Dataset",
        ["Hospital A – Patients", "Hospital B – Appointments"]
    )

    # -------------------------------------------------
    # LOAD DATA
    # -------------------------------------------------
    if hospital == "Hospital A – Patients":
        df = pd.read_csv("patients_updated.csv")
    else:
        df = pd.read_csv("appointments_updated.csv")

    # -------------------------------------------------
    # PREPROCESSING
    # -------------------------------------------------
    for col in df.columns:
        if "date" in col.lower():
            df[col] = pd.to_datetime(df[col], errors="coerce")

    df.drop_duplicates(inplace=True)
    df.fillna(method="ffill", inplace=True)

    # -------------------------------------------------
    # FILTERS
    # -------------------------------------------------
    st.sidebar.markdown("### 🔎 Filters")

    if "gender" in df.columns:
        gender_filter = st.sidebar.multiselect(
            "Gender",
            df["gender"].unique(),
            default=df["gender"].unique()
        )
        df = df[df["gender"].isin(gender_filter)]

    if "bed_availability" in df.columns:
        bed_filter = st.sidebar.multiselect(
            "Bed Availability",
            df["bed_availability"].unique(),
            default=df["bed_availability"].unique()
        )
        df = df[df["bed_availability"].isin(bed_filter)]

    # -------------------------------------------------
    # KPI METRICS
    # -------------------------------------------------
    st.markdown("## 📌 Key Metrics")

    k1, k2, k3 = st.columns(3)

    k1.metric("Total Records", len(df))

    if "age" in df.columns:
        k2.metric("Average Age", round(df["age"].mean(), 1))
    else:
        k2.metric("Average Age", "N/A")

    if "bed_availability" in df.columns:
        k3.metric(
            "Beds Available (%)",
            round((df["bed_availability"] == "Available").mean() * 100, 1)
        )

    # -------------------------------------------------
    # DATA PREVIEW
    # -------------------------------------------------
    with st.expander("📄 View Cleaned Dataset"):
        st.dataframe(df.head(25))

    # -------------------------------------------------
    # PATIENTS DASHBOARD
    # -------------------------------------------------
    if hospital == "Hospital A – Patients":
        st.header("🧑‍⚕️ Patient Analytics")

        if "department" in df.columns:
            fig = px.bar(
                df,
                x="department",
                color="bed_availability",
                title="Patients by Department & Bed Availability"
            )
            st.plotly_chart(fig, use_container_width=True)

        if "service" in df.columns:
            fig = px.pie(
                df,
                names="service",
                title="Service-wise Patient Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        if "arrival_date" in df.columns and "departure_date" in df.columns:
            df["length_of_stay"] = (
                df["departure_date"] - df["arrival_date"]
            ).dt.days

            fig = px.box(
                df,
                x="department",
                y="length_of_stay",
                title="Length of Stay by Department"
            )
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # APPOINTMENTS DASHBOARD
    # -------------------------------------------------
    else:
        st.header("📅 Appointment Analytics")

        if "status" in df.columns:
            fig = px.bar(
                df,
                x="status",
                color="bed_availability",
                title="Appointment Status vs Bed Availability"
            )
            st.plotly_chart(fig, use_container_width=True)

        if "department" in df.columns:
            fig = px.bar(
                df,
                x="department",
                color="gender",
                title="Appointments by Department & Gender"
            )
            st.plotly_chart(fig, use_container_width=True)

        if "discharge_date" in df.columns:
            fig = px.histogram(
                df,
                x="discharge_date",
                title="Discharge Date Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

    # -------------------------------------------------
    # FOOTER
    # -------------------------------------------------
    st.markdown("---")
    st.markdown(
        "**Major Project – Hospital Analytics System**  \n"
        "🔹 Secure Login  🔹 Interactive Charts  🔹 Real-world Hospital KPIs"
    )


