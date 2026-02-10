# ===================== IMPORTS =====================
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
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

# ================= SESSION STATE =================
if "logged_in" not in st.session_state:
    st.session_state["logged_in"] = False

if "hospital" not in st.session_state:
    st.session_state["hospital"] = None

# ===================== LOGIN PAGE =====================
def login_page():
    # 1. Custom CSS for a Premium Healthcare Feel
    st.markdown("""
        <style>
        /* Background and Global Styles */
        .stApp {
            background: linear-gradient(to right, #ffffff, #f0f9ff);
        }
        
        /* Centering the Login Container */
        div.block-container {
            padding-top: 5rem;
            max-width: 800px;
        }

        /* Form Styling */
        [data-testid="stForm"] {
            border: none;
            padding: 40px;
            border-radius: 15px;
            background-color: white;
            box-shadow: 0 10px 25px rgba(0,0,0,0.05);
        }

        /* Input Field Focus Colors */
        input:focus {
            border-color: #10b981 !important;
            box-shadow: 0 0 0 0.2rem rgba(16, 185, 129, 0.25) !important;
        }

        /* Login Button Styling */
        div.stButton > button:first-child {
            background-color: #10b981;
            color: white;
            border: none;
            padding: 0.6rem 2rem;
            font-weight: bold;
            border-radius: 8px;
            width: 100%;
            transition: all 0.3s ease;
        }

        div.stButton > button:hover {
            background-color: #059669;
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(16, 185, 129, 0.3);
        }

        /* Title and Subtitle */
        .main-title {
            color: #0f172a;
            font-size: 36px;
            font-weight: 800;
            margin-bottom: 5px;
        }
        .sub-title {
            color: #64748b;
            font-size: 16px;
            margin-bottom: 30px;
        }
        </style>
    """, unsafe_allow_html=True)

    # 2. Layout with a Graphic Image
    col1, col2 = st.columns([1, 1.2], gap="large")

    with col1:
        # High-quality Healthcare Graphic
        st.image("https://img.freepik.com/free-vector/doctors-concept-illustration_114360-1515.jpg", 
                 caption="Data-Driven Healthcare Solutions")
        st.markdown("<h2 class='main-title'>Hospital Analytics</h2>", unsafe_allow_html=True)
        st.markdown("<p class='sub-title'>Predicting bed occupancy and analyzing patient trends with precision.</p>", unsafe_allow_html=True)

    with col2:
        # The actual Login Form
        with st.form("login_form"):
            st.markdown("### Secure Login")
            username = st.text_input("Username", placeholder="admin")
            password = st.text_input("Password", type="password", placeholder="••••••••")
            hospital = st.selectbox("Assigning Hospital", ["Hospital1", "Hospital2"])
            
            st.markdown("<br>", unsafe_allow_html=True)
            login_btn = st.form_submit_button("Access Dashboard")

        if login_btn:
            if username == "admin" and password == "admin123":
                st.session_state["logged_in"] = True
                st.session_state["hospital"] = hospital
                st.rerun()
            else:
                st.error("Invalid Credentials. Please check and try again.")
    
# show login if not logged in
if not st.session_state["logged_in"]:
    login_page()
    st.stop()
# ===================== LOAD DATA =====================
if st.session_state.hospital == "Hospital1":
    df = pd.read_csv("appointments_cleaned.csv")   # hospital1 → appointments
    table_name = "appointments"
else:
    df = pd.read_csv("patients_cleaned.csv")       # hospital2 → patients
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
    ["Overview", "Visualizations", "Correlation", "Forecasting", "Database"]
)

st.sidebar.markdown("---")
st.sidebar.write("Hospital:", st.session_state["hospital"])

# Logout button
if st.sidebar.button("Logout"):
    st.session_state["logged_in"] = False
    st.session_state["hospital"] = None
    st.rerun()

# ===================== CLEAN COLUMNS FOR CHARTS =====================
ignore_cols = [c for c in df.columns if "id" in c.lower() or "date" in c.lower()]
num_cols = [c for c in df.select_dtypes(include=np.number).columns if c not in ignore_cols]
cat_cols = [c for c in df.select_dtypes(include="object").columns if c not in ignore_cols]

# ===================== DASHBOARD =====================
if page == "Overview":

    st.title("Hospital Overview")

    # ================= HOSPITAL 1 =================
    if st.session_state.hospital == "Hospital1":

        st.subheader("Appointment Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Appointments", len(df))
        c2.metric("Departments", df["department"].nunique() if "department" in df.columns else "N/A")
        c3.metric("Completed", df["status"].eq("Completed").sum() if "status" in df.columns else "N/A")
        c4.metric("Beds Available", df["bed_availability"].sum() if "bed_availability" in df.columns else "N/A")


        col1, col2 = st.columns(2)

        # Department Bar Chart
        if "department" in df.columns:
            with col1:
                fig, ax = plt.subplots()
                df["department"].value_counts().plot.bar(ax=ax)
                ax.set_title("Appointments by Department")
                st.pyplot(fig)

        # Status Pie Chart
        if "status" in df.columns:
            with col2:
                fig, ax = plt.subplots()
                df["status"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
                ax.set_ylabel("")
                ax.set_title("Appointment Status")
                st.pyplot(fig)


    # ================= HOSPITAL 2 =================
    else:

        st.subheader("Patient Overview")

        c1, c2, c3, c4 = st.columns(4)

        c1.metric("Total Patients", len(df))
        c2.metric("Average Age", round(df["age"].mean(),1) if "age" in df.columns else "N/A")
        c3.metric("Departments", df["department"].nunique() if "department" in df.columns else "N/A")
        c4.metric("Beds Available", df["bed_availability"].sum() if "bed_availability" in df.columns else "N/A")


        col1, col2 = st.columns(2)

        # Gender Bar Chart
        if "gender" in df.columns:
            with col1:
                fig, ax = plt.subplots()
                df["gender"].value_counts().plot.bar(ax=ax)
                ax.set_title("Gender Distribution")
                st.pyplot(fig)

        # Department Pie Chart
        if "department" in df.columns:
            with col2:
                fig, ax = plt.subplots()
                df["department"].value_counts().plot.pie(autopct="%1.1f%%", ax=ax)
                ax.set_ylabel("")
                ax.set_title("Department Distribution")
                st.pyplot(fig)
    st.markdown("---")
    st.subheader("Dataset Overview")

    clean_df = df.dropna()

    col1, col2 = st.columns(2)

    with col1:
        st.write("Dataset Shape:", df.shape)
    with col2:
        st.write("After Cleaning:", clean_df.shape)

    with st.expander("View Cleaned Data"):
        st.dataframe(clean_df.head())

    with st.expander("Statistical Summary"):
        st.dataframe(clean_df.describe())

# ===================== VISUALIZATIONS =====================
elif page == "Visualizations":

    st.title("Hospital Analytics Visualizations")

    # ================= PATIENT DATA =================
    if st.session_state.hospital == "Hospital2":

        col1, col2 = st.columns(2)

        # Gender Bar Chart
        with col1:
            fig = px.bar(
                df,
                x="gender",
                title="Gender Distribution",
                color="gender"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Department Pie Chart
        with col2:
            fig = px.pie(
                df,
                names="department",
                title="Department Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Age Distribution
        col3, col4 = st.columns(2)

        with col3:
            fig = px.histogram(
                df,
                x="age",
                nbins=30,
                title="Age Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Service Graph
        if "service" in df.columns:
            with col4:
                fig = px.bar(
                    df["service"].value_counts().reset_index(),
                    x="service",
                    y="count",
                    title="Hospital Services Usage"
                )
                st.plotly_chart(fig, use_container_width=True)

        # Length of Stay vs Age (VERY PROFESSIONAL)
        if "length_of_stay" in df.columns:
            fig = px.scatter(
                df,
                x="age",
                y="length_of_stay",
                color="gender",
                title="Age vs Length of Stay"
            )
            st.plotly_chart(fig, use_container_width=True)

    # ================= APPOINTMENT DATA =================
    else:

        col1, col2 = st.columns(2)

        # Department Bar Chart
        with col1:
            fig = px.bar(
                df,
                x="department",
                title="Appointments by Department",
                color="department"
            )
            st.plotly_chart(fig, use_container_width=True)

        # Status Pie Chart
        with col2:
            fig = px.pie(
                df,
                names="status",
                title="Appointment Status"
            )
            st.plotly_chart(fig, use_container_width=True)

        col3, col4 = st.columns(2)

        # Appointment Index vs Age
        with col3:
            fig = px.scatter(
                df,
                x=df.index,
                y="age",
                color="status",
                title="Appointment Index vs Age",
                labels={"x": "Appointment Index"}
            )
            st.plotly_chart(fig, use_container_width=True)

        # Daily Appointment Trend (VERY IMPORTANT)
        date_cols = [c for c in df.columns if "date" in c.lower()]

        if date_cols:
            date_col = date_cols[0]

            df[date_col] = pd.to_datetime(df[date_col])

            daily = df.groupby(date_col).size().reset_index(name="appointments")

            with col4:
                fig = px.line(
                    daily,
                    x=date_col,
                    y="appointments",
                    title="Daily Appointment Trend"
                )
                st.plotly_chart(fig, use_container_width=True)
# ===================== CORRELATION =====================
elif page == "Correlation":

    st.title("Correlation Matrix")

    corr = df.select_dtypes(include=np.number).corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        aspect="auto",
        title="Correlation Heatmap"
    )

    st.plotly_chart(fig, use_container_width=True)

# ===================== FORECASTING =====================
elif page == "Forecasting":

    st.title("Daily Admissions Forecast")

    # Automatically detect date column
    date_cols = [c for c in df.columns if "date" in c.lower()]

    if not date_cols:
        st.warning("No date column found for forecasting.")
        st.stop()

    date_col = date_cols[0]

    # Convert to datetime
    df[date_col] = pd.to_datetime(df[date_col])

    # Create daily count (VERY IMPORTANT for hospital analytics)
    daily = df.groupby(date_col).size().reset_index(name="admissions")

    if len(daily) < 20:
        st.warning("Not enough data for forecasting.")
        st.stop()

    # ================= ARIMA =================
    model = ARIMA(daily["admissions"], order=(1,1,1))
    model_fit = model.fit()

    forecast_steps = st.slider(
    "Select Forecast Days",
    min_value=7,
    max_value=30,
    value=10
    )
    forecast = model_fit.forecast(steps=forecast_steps)

    # Future dates
    future_dates = pd.date_range(
        start=daily[date_col].max(),
        periods=forecast_steps+1,
        freq="D"
    )[1:]

    # ================= INTERACTIVE PLOT =================
    fig = go.Figure()

    # Actual line
    fig.add_trace(go.Scatter(
        x=daily[date_col],
        y=daily["admissions"],
        mode='lines',
        name='Actual Admissions'
    ))

    # Forecast line
    fig.add_trace(go.Scatter(
        x=future_dates,
        y=forecast,
        mode='lines+markers',
        name='Forecast',
        line=dict(dash='dash')
    ))

    fig.update_layout(
        title="Daily Patient Admission Forecast",
        xaxis_title="Date",
        yaxis_title="Number of Admissions",
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    # Show forecast values
    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted Admissions": forecast.round(0)
    })

    st.subheader("Next 10 Days Prediction")
    st.dataframe(forecast_df)

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

























